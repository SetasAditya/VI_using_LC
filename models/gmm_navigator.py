"""
GMM Navigator.

GRU-based controller that takes sense features + topology diagnostics
and outputs transport parameters (dt_scale, gamma, M_diag, G, u).

Architecture:
    Input:   feat = [nu_feats | dF | topo_feats]  (fixed-size vector)
    GRU:     h_k = GRU(feat_k, h_{k-1})
    Heads:   dt_scale, gamma, M_diag, G, u from MLP heads on h_k

Output parameterization (all constrained):
    dt_scale   = sigmoid(h) * (dt_scale_max - dt_scale_min) + dt_scale_min
                 target-neutral timestep multiplier — scales mixing speed only
    gamma      = softplus(h) + gamma_min         (friction > 0)
    M_diag     = softplus(h) + M_min             ([phi_dim] positive diagonal masses)
    G          = linear(h) reshaped [phi_dim, port_rank]  (Gram-Schmidt normalized)
    gain       = u_max * tanh(h)                 (scalar steering intensity, stored as u [C_max])

Mode B control (per-particle):
    The navigator outputs per-cluster amplitudes u [C_max].
    compute_per_particle_Gu() converts these into a per-particle Gu [M, phi_dim]
    by steering each particle toward the minimum-mass cluster centroid, scaled
    by u[c_m] — the amplitude for that particle's own cluster.  This means
    heavy-cluster particles get steered strongly toward sparse modes, while
    sparse-cluster particles get small or zero perturbation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from topology.diagnostics import TopoDiagnostics


# ──────────────────────────────────────────────────────────────────────────────
# Transport parameter container
# ──────────────────────────────────────────────────────────────────────────────

class TransportParams:
    """Container for navigator outputs (transport parameters)."""
    def __init__(
        self,
        dt_scale: torch.Tensor,  # scalar — effective timestep multiplier (target-neutral)
        gamma: torch.Tensor,     # scalar — friction coefficient
        M_diag: torch.Tensor,    # [phi_dim] diagonal mass matrix
        G: torch.Tensor,         # [phi_dim, port_rank] port basis (Gram-Schmidt orthonormal)
        u: torch.Tensor,         # [C_max] expand of scalar gain — u[0] is the gain used by compute_per_particle_Gu
    ):
        self.dt_scale = dt_scale
        self.gamma = gamma
        self.M_diag = M_diag
        self.G = G
        self.u = u


# ──────────────────────────────────────────────────────────────────────────────
# GRU Navigator
# ──────────────────────────────────────────────────────────────────────────────

class GMMNavigator(nn.Module):
    """
    GRU-based navigator for GMM SPHS transport.

    Maintains a hidden belief state h across streaming batches.
    At each batch: h_k = GRU(feat_k, h_{k-1}), then MLP heads → Theta_k.
    """

    def __init__(
        self,
        feat_dim: int,
        gru_hidden: int,
        phi_dim: int,
        port_rank: int = 8,
        C_max: int = 8,            # must match topology C_max; u head outputs [C_max]
        M_type: str = "diagonal",  # "diagonal" or "scalar"
        dt_scale_min: float = 0.3,
        dt_scale_max: float = 3.0,
        gamma_min: float = 0.1,
        gamma_max: float = 5.0,
        u_max: float = 2.0,
        mlp_hidden: int = 256,
        control_mode: str = "A",   # A: no control, B: per-cluster steering + Girsanov
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.gru_hidden = gru_hidden
        self.phi_dim = phi_dim
        self.port_rank = port_rank
        self.C_max = C_max
        self.M_type = M_type
        self.dt_scale_min = dt_scale_min
        self.dt_scale_max = dt_scale_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.u_max = u_max
        self.control_mode = control_mode

        # ── Input normalisation ───────────────────────────────────────────────
        # LayerNorm stabilises feature scale across episodes with different K,
        # preventing the GRU from settling into a near-zero activation that
        # produces constant navigator outputs (the "lazy constant" failure mode).
        self.feat_norm = nn.LayerNorm(feat_dim)

        # ── GRU core ──────────────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        # ── Output heads ──────────────────────────────────────────────────────
        def mlp_head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(gru_hidden),    # normalise GRU output before heads
                nn.Linear(gru_hidden, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, out_dim),
            )

        # dt_scale: effective timestep multiplier — controls mixing speed, NOT target
        # sigmoid output maps to [dt_scale_min, dt_scale_max], always positive
        self.dt_scale_head = mlp_head(1)

        # Gamma: friction coefficient (scalar)
        self.gamma_head = mlp_head(1)

        # M_diag: diagonal mass [phi_dim]  (or scalar)
        if M_type == "diagonal":
            self.M_head = mlp_head(phi_dim)
        else:  # scalar
            self.M_head = mlp_head(1)

        # G: port basis [phi_dim * port_rank]
        # gain: single scalar steering intensity for mode B.
        # Amplitude per cluster is computed deterministically as gain * W_c[c]
        # in compute_per_particle_Gu, which is permutation-equivariant —
        # the navigator doesn't need to learn which slot index is heavy.
        if control_mode in ("B", "C"):
            self.G_head    = mlp_head(phi_dim * port_rank)
            self.gain_head = mlp_head(1)   # scalar gain — replaces [C_max] u_head
        else:
            self.G_head    = None
            self.gain_head = None

        # ── Init: moderate weights so heads are sensitive to features ─────────
        # 1e-3 was too small — GRU hidden state was ignored; beta≈const always.
        # 1e-2 gives initial output variation ~0.1 sigma, enough to learn from.
        for head in [self.dt_scale_head, self.gamma_head, self.M_head]:
            # Only last Linear layer; LayerNorm before it handles scale
            nn.init.normal_(head[-1].weight, 0, 1e-2)
            nn.init.zeros_(head[-1].bias)

    def forward_step(
        self,
        feat: torch.Tensor,          # [feat_dim]
        h: Optional[torch.Tensor],   # [1, 1, gru_hidden] GRU hidden state
    ) -> Tuple[TransportParams, torch.Tensor]:
        """
        Single streaming step.

        Args:
            feat: [feat_dim] or [1, feat_dim] feature vector for this batch
            h:    [1, 1, gru_hidden] previous hidden state (None = zeros)
        Returns:
            (TransportParams, h_new)
        """
        # Normalise input features and ensure shape [1, 1, feat_dim]
        x = self.feat_norm(feat.view(1, -1)).view(1, 1, -1)

        # GRU update
        if h is None:
            out, h_new = self.gru(x)
        else:
            out, h_new = self.gru(x, h)

        h_vec = h_new.squeeze(0).squeeze(0)  # [gru_hidden]

        # ── dt_scale: timestep multiplier ─────────────────────────────────────
        # Sigmoid maps to [dt_scale_min, dt_scale_max].
        # Unlike beta, this does NOT change the BAOAB stationary distribution —
        # it scales both drift AND noise proportionally, so the target p(phi)
        # is unchanged; only mixing speed (exploration rate) changes.
        dt_scale_raw = self.dt_scale_head(h_vec).squeeze()
        rng = self.dt_scale_max - self.dt_scale_min
        dt_scale = torch.sigmoid(dt_scale_raw) * rng + self.dt_scale_min

        # ── Gamma ────────────────────────────────────────────────────────────
        gamma_raw = self.gamma_head(h_vec).squeeze()
        gamma = torch.clamp(
            F.softplus(gamma_raw) + self.gamma_min,
            max=self.gamma_max,
        )

        # ── M_diag ───────────────────────────────────────────────────────────
        M_raw = self.M_head(h_vec)
        if self.M_type == "diagonal":
            M_diag = F.softplus(M_raw) + 0.1  # [phi_dim] all positive
        else:
            M_scalar = F.softplus(M_raw).squeeze() + 0.1
            M_diag = M_scalar.expand(self.phi_dim)

        # ── G and gain ───────────────────────────────────────────────────────
        if self.control_mode in ("B", "C") and self.G_head is not None:
            G_raw  = self.G_head(h_vec).view(self.phi_dim, self.port_rank)
            G      = _gram_schmidt(G_raw)                    # [phi_dim, port_rank]
            gain   = self.u_max * torch.tanh(self.gain_head(h_vec).squeeze())  # scalar
            # u stored as [C_max] of identical gain for API compat with TransportParams;
            # compute_per_particle_Gu only reads u[0] as the gain scalar.
            u = gain.expand(self.C_max)
        else:
            G = torch.zeros(self.phi_dim, self.port_rank, device=feat.device)
            u = torch.zeros(self.C_max, device=feat.device)

        params = TransportParams(dt_scale=dt_scale, gamma=gamma, M_diag=M_diag, G=G, u=u)
        return params, h_new

    def forward_episode(
        self,
        feat_sequence: List[torch.Tensor],  # list of T [feat_dim] tensors
    ) -> Tuple[List[TransportParams], List[torch.Tensor]]:
        """
        Unrolled forward pass over an entire episode (for training).

        Args:
            feat_sequence: T feature vectors, one per batch
        Returns:
            (list of T TransportParams, list of T hidden states)
        """
        h = None
        params_list = []
        h_list = []

        for feat in feat_sequence:
            params, h = self.forward_step(feat, h)
            params_list.append(params)
            h_list.append(h)

        return params_list, h_list

    def topology_adjustment(
        self,
        params: TransportParams,
        topo: TopoDiagnostics,
        C_target: int,
        ESS_min: float,
    ) -> TransportParams:
        """
        Hard-coded topology-aware adjustments (applied after navigator output).

        Rule-based corrections:
            - C_tau < C_target: increase dt_scale (explore), reduce gamma (less friction)
            - min_ESS_c < ESS_min: boost u amplitudes for under-sampled clusters

        u is now [C_max] per-cluster; boosting all amplitudes still increases
        the overall steering magnitude. Per-cluster selective boosting could be
        added here if C_tau and ESS_c per-cluster are tracked.
        """
        dt_scale = params.dt_scale
        gamma    = params.gamma
        u        = params.u  # [C_max]

        if topo.C_tau < C_target:
            # Mode collapse: explore — larger steps to escape local minima
            dt_scale = (dt_scale * 0.5).clamp(max=self.dt_scale_max)
            gamma    = gamma * 0.7

        if topo.min_ESS_c < ESS_min and topo.C_tau > 1:
            # Under-sampled basin: boost all cluster amplitudes
            # (heavy-cluster particles will be steered more strongly toward sparse modes)
            u = (u * 1.5).clamp(-self.u_max, self.u_max)

        return TransportParams(
            dt_scale=dt_scale, gamma=gamma, M_diag=params.M_diag, G=params.G, u=u
        )


# ──────────────────────────────────────────────────────────────────────────────
# Feature builder
# ──────────────────────────────────────────────────────────────────────────────

def build_features(
    nu: torch.Tensor,          # [K + K*D] sufficient stat residual
    dF: torch.Tensor,          # scalar free energy change
    topo: TopoDiagnostics,     # topology diagnostics
    tau_quantiles: List[float],
    target_feat_dim: int,
) -> torch.Tensor:
    """
    Build fixed-size feature vector for navigator input.

    Format: [nu_feats | dF | topo_feats] padded/truncated to target_feat_dim.

    Args:
        nu:              [K + K*D] sufficient stat residual
        dF:              scalar
        topo:            TopoDiagnostics
        tau_quantiles:   list of quantile fractions
        target_feat_dim: desired output dimension
    Returns:
        feat: [target_feat_dim]
    """
    device = nu.device

    # Collect components
    parts = [
        nu.float(),                                         # [K + K*D]
        dF.view(1).float(),                                 # [1]
        topo.to_feature_vector(tau_quantiles).float(),      # [topo feat_dim]
    ]

    feat_raw = torch.cat(parts)  # variable size

    # Pad or truncate to target_feat_dim
    current_dim = feat_raw.shape[0]
    if current_dim < target_feat_dim:
        feat = F.pad(feat_raw, (0, target_feat_dim - current_dim))
    else:
        feat = feat_raw[:target_feat_dim]

    # Replace NaN/inf
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Per-particle control computation (mode B)
# ──────────────────────────────────────────────────────────────────────────────

def compute_per_particle_Gu(
    phi: torch.Tensor,            # [M, phi_dim]
    assignments: torch.Tensor,    # [M] cluster ids ∈ {0,...,C-1}, -1 = unassigned
    G: torch.Tensor,              # [phi_dim, port_rank] orthonormal port basis
    u_per_cluster: torch.Tensor,  # [C_max] — u[0] is the scalar gain (all entries equal)
    W_c: torch.Tensor,            # [C_max] cluster masses from topo diagnostics
) -> torch.Tensor:                # [M, phi_dim]
    """
    Compute per-particle Gu_m for mode B transport.

    Amplitude design (permutation-equivariant):
        amp_m = gain * W_c[c_m]

    where gain = u_per_cluster[0] (a single learned scalar) and W_c[c_m] is the
    mass of particle m's own cluster.  Heavy-cluster particles get large kicks;
    sparse-cluster particles (already in the under-sampled basin) get near-zero
    kicks and are left undisturbed.

    All non-target particles are steered toward the centroid of the minimum-mass
    cluster (the most under-represented mode).  Particles already IN the target
    cluster get amp=0 — they don't need steering and the Girsanov correction for
    them would be pure noise.

    Permutation equivariance: amplitudes depend on W_c values (semantically
    meaningful) not slot indices (arbitrary across episodes), so the navigator
    doesn't need to learn a per-permutation mapping.
    """
    M, phi_dim = phi.shape
    device = phi.device
    Gu = torch.zeros(M, phi_dim, device=device, dtype=phi.dtype)

    valid_mask = assignments >= 0
    if not valid_mask.any():
        return Gu

    active_ids = assignments[valid_mask].unique().tolist()
    if len(active_ids) < 2:
        return Gu  # single cluster — nothing to steer toward

    # ── Scalar gain from navigator (same for all clusters) ───────────────────
    # u_per_cluster is expand(C_max) of a single gain scalar; read [0].
    gain = u_per_cluster[0]  # differentiable scalar; gradient flows back through gain_head

    # ── Per-cluster amplitude: gain * W_c[c]  (detach masses — not differentiable) ──
    # Heavy cluster → large amp → particles steered hard toward sparse mode.
    # Sparse cluster → small amp → particles barely perturbed.
    W_c_det = W_c.detach()  # [C_max] — stop grad through cluster masses
    amp_per_cluster = gain * W_c_det  # [C_max], grad flows through gain only

    # ── Target cluster = minimum-mass active cluster ──────────────────────────
    active_masses = {c: float(W_c_det[c].item()) if c < len(W_c_det) else 0.0
                     for c in active_ids}
    target_c = min(active_masses, key=active_masses.get)

    # ── Centroid of target cluster (detached — no grad through particle positions) ──
    target_mask = assignments == target_c
    if not target_mask.any():
        return Gu
    target_centroid = phi[target_mask].detach().mean(0)  # [phi_dim]

    # ── Projection matrix onto G's column space ──────────────────────────────
    GGT = G @ G.t()  # [phi_dim, phi_dim]

    # ── Per-particle direction toward target centroid ─────────────────────────
    directions = target_centroid.unsqueeze(0) - phi.detach()   # [M, phi_dim]
    norms      = directions.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    directions = directions / norms                             # unit vectors
    projected  = directions @ GGT                              # [M, phi_dim]

    # ── Assemble amplitudes: gain * W_c[c_m], zero for target cluster ─────────
    amps = torch.zeros(M, device=device, dtype=gain.dtype)
    for c in active_ids:
        if c == target_c:
            continue   # particles already in sparse mode — leave them alone
        if c >= len(amp_per_cluster):
            continue
        mask_c = assignments == c
        amps[mask_c] = amp_per_cluster[c]  # same amp for all particles in cluster c

    Gu = amps.unsqueeze(-1) * projected    # [M, phi_dim], grad through amps → gain

    return Gu


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gram_schmidt(B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Orthonormalize columns of B via Modified Gram-Schmidt.

    Args:
        B: [d, r]
    Returns:
        Q: [d, r] orthonormal columns
    """
    d, r = B.shape
    B = torch.nan_to_num(B, nan=0.0)
    Q_cols = []

    for k in range(r):
        v = B[:, k].clone()
        for q in Q_cols:
            v = v - (q @ v) * q
        nrm = v.norm()
        if nrm < eps:
            v = torch.randn_like(v)
            nrm = v.norm()
        Q_cols.append(v / nrm.clamp(min=eps))

    return torch.stack(Q_cols, dim=1)  # [d, r]