"""
GMM Embedder.

Maps particle state phi → fixed-size embedding z for PHC.

The embedding must be:
    1. Fixed-size (for KDE and filtration)
    2. Applied AFTER canonicalization (so permutation symmetry is removed)
    3. Normalized per block (so KDE bandwidth is meaningful)
"""
from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn

from data.gmm_problem import unpack_phi, L_vec_to_matrix


class GMMEmbedder(nn.Module):
    """
    Maps phi → embedding z for PHC topology detection.

    Three embedding modes (selected by arguments to embed()):

    (A) param  [default, no X_ref]: [mu_flat | log_diag_L | pi]
        Dimension: K*D + K*D + K.
        After BAOAB convergence all particles cluster in one parameter
        mode → C_tau=1. Correct for a unimodal posterior; useless for
        multi-basin topology detection.

    (B) combined [X_ref, resp_only=False]: appends mean responsibilities
        r^(m) ∈ [0,1]^K to the param block.
        Dimension: K*D + K*D + K + K.
        Still high-dimensional; responsibilities are diluted by param block.

    (C) resp_only [X_ref, resp_only=True]:  ONLY mean responsibilities.
        Dimension: K.
        K=4 modes map to near-orthogonal corners of Δ^{K-1}.  Silverman
        bandwidth in K-dim space is much tighter → PHC sees distinct basins.
        THIS IS THE MODE THAT ACTUALLY WORKS for multi-basin detection.
    """

    def __init__(self, K: int, D: int):
        super().__init__()
        self.K = K
        self.D = D
        self.embed_dim      = K * D + K * D + K        # param mode
        self.embed_dim_resp = K * D + K * D + K + K    # combined mode
        self.embed_dim_resp_only = K                   # resp_only mode

    def embed(
        self,
        phi: torch.Tensor,                        # [M, phi_dim]
        X_ref: Optional[torch.Tensor] = None,     # [B, D] reference batch
        resp_only: bool = False,                  # return only K-dim responsibilities
        mu_only: bool = False,                    # return only K*D-dim means (RECOMMENDED for PHC)
    ) -> torch.Tensor:
        """
        Map particles to embedding space.

        Args:
            phi:       [M, phi_dim]
            X_ref:     [B, D] optional reference batch
            resp_only: return [M, K] mean responsibilities. WRONG for PHC:
                       well-fit particles all land at simplex center (1/K, ..., 1/K).
            mu_only:   return [M, K*D] canonicalized means ONLY. CORRECT for PHC:
                       different hypotheses give different mu vectors; K*D=8 for K=4,D=2
                       → Silverman bandwidth is tight enough to resolve distinct modes.
        Returns:
            z: [M, embed_dim*] — shape depends on mode
        """
        M = phi.shape[0]
        K, D = self.K, self.D

        pi_tilde, mu, L_vecs = unpack_phi(phi, K, D)
        pi = torch.softmax(pi_tilde, dim=-1)   # [M, K]

        # ── mu_only mode: canonicalized means only [M, K*D] ──────────────────
        # Best embedding for PHC: captures which data region each component
        # is assigned to. Different parameter hypotheses → different mu vectors.
        # Dimension K*D is low enough for Silverman to give a meaningful σ.
        if mu_only:
            mu_flat = mu.reshape(M, K * D)
            return self._normalize(mu_flat, M)   # [M, K*D]

        # ── resp_only mode: return just K-dim responsibilities ────────────────
        # NOTE: DO NOT USE FOR PHC. Correct particles all map to (1/K,...,1/K).
        if resp_only and X_ref is not None and X_ref.shape[0] > 0:
            L = L_vec_to_matrix(L_vecs, D)
            resp = self._mean_responsibilities(phi, X_ref, pi, mu, L, K, D, M)
            return self._normalize(resp, M)   # [M, K]

        # ── param + optional responsibility blocks ────────────────────────────
        mu_flat    = mu.reshape(M, K * D)
        L          = L_vec_to_matrix(L_vecs, D)
        L_diag     = L.diagonal(dim1=-2, dim2=-1)
        log_L_flat = torch.log(L_diag.abs().clamp(min=1e-6)).reshape(M, K * D)

        parts = [mu_flat, log_L_flat, pi]

        if X_ref is not None and X_ref.shape[0] > 0:
            resp = self._mean_responsibilities(phi, X_ref, pi, mu, L, K, D, M)
            parts.append(resp)

        z_raw = torch.cat(parts, dim=-1)
        return self._normalize(z_raw, M)

    def _mean_responsibilities(
        self,
        phi: torch.Tensor,
        X_ref: torch.Tensor,     # [B, D]
        pi: torch.Tensor,        # [M, K]
        mu: torch.Tensor,        # [M, K, D]
        L: torch.Tensor,         # [M, K, D, D]
        K: int, D: int, M: int,
    ) -> torch.Tensor:
        """
        Compute mean soft responsibility per component over X_ref.

        r_k^(m) = (1/B) sum_b P(z_b = k | x_b, phi^(m))

        Returns: [M, K]
        """
        import math as _math
        B = X_ref.shape[0]

        # diff: [M, K, B, D]
        diff = X_ref[None, None, :, :] - mu[:, :, None, :]

        # Solve L @ alpha = diff  →  Mahalanobis
        MK = M * K
        L_flat = L.reshape(MK, D, D)
        diff_flat = diff.reshape(MK, B, D).permute(0, 2, 1)   # [MK, D, B]
        alpha = torch.linalg.solve_triangular(L_flat, diff_flat, upper=False)
        maha = (alpha ** 2).sum(1).reshape(M, K, B)            # [M, K, B]

        log_det = 2.0 * L_flat.diagonal(dim1=-2, dim2=-1).abs().clamp(min=1e-8).log().sum(-1)
        log_det = log_det.reshape(M, K)                         # [M, K]

        log_norm = -0.5 * (D * _math.log(2 * _math.pi) + log_det[:, :, None])
        log_gauss = log_norm - 0.5 * maha                       # [M, K, B]

        log_pi = torch.log(pi.clamp(min=1e-8))                  # [M, K]
        log_joint = log_pi[:, :, None] + log_gauss              # [M, K, B]

        # Responsibilities: softmax over K dim
        log_resp = log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)  # [M, K, B]
        resp = log_resp.exp()                                    # [M, K, B]

        return resp.mean(dim=-1)                                 # [M, K] mean over B

    def _normalize(self, z: torch.Tensor, M: int) -> torch.Tensor:
        """Per-feature standardization across the M particles."""
        if M > 1:
            mean = z.mean(dim=0, keepdim=True)
            std  = z.std(dim=0, keepdim=True).clamp(min=1e-6)
            return (z - mean) / std
        return z

    def forward(
        self,
        phi: torch.Tensor,
        X_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.embed(phi, X_ref)