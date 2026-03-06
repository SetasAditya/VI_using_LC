"""
BAOAB Integrator for GMM phi space.

Wraps the core BAOABIntegrator from ph_inference, adapting it for:
    - phi-space (not z-space from the PDE codebase)
    - diagonal mass matrix M_diag (from navigator)
    - SMC weight update after transport
    - Resampling trigger (ESS-based)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ── Inline BAOAB (self-contained, adapted from ph_inference/dynamics/sphs.py) ──

class BAOABIntegrator(nn.Module):
    """
    BAOAB integrator for underdamped Langevin dynamics in phi-space.

    Convention for beta semantics
    ─────────────────────────────
    The target distribution is exp(-beta * U(phi)).  We implement this by
    scaling the gradient: force = -beta * nabla_U.  Momentum and noise use
    the unscaled mass M so that the invariant measure of the full system is:

        pi(phi, p) ∝ exp(-beta * U(phi)) * exp(-1/2 p^T M^{-1} p)

    This is the "scaled-potential" convention, which is internally consistent
    with the SMC weight update  log w += -beta * U.

    All of beta, gamma, M_diag may be 0-dim tensors so that gradients flow
    from navigator parameters through the BAOAB rollout into phi_final.
    """

    def __init__(
        self,
        phi_dim: int,
        mass: float = 1.0,
        friction: float = 1.0,
        dt: float = 0.01,
    ):
        super().__init__()
        self.phi_dim = phi_dim
        self.default_mass = mass
        self.default_friction = friction
        self.dt = dt

    def _ou_coeffs(
        self,
        friction,           # float OR 0-dim tensor
        dt: float,
        mass_diag: torch.Tensor,  # [phi_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable O-U coefficients.

        c1 = exp(-gamma * h)            [0-dim tensor]
        c2 = sqrt(1 - c1^2) * sqrt(M)  [phi_dim tensor]

        Uses torch.exp so gradients flow when gamma is a tensor.
        """
        if torch.is_tensor(friction):
            gamma_h = friction * dt
            c1 = torch.exp(-gamma_h)                          # 0-dim tensor
            c2 = torch.sqrt((1 - c1 ** 2).clamp(min=0)) * mass_diag.sqrt()
        else:
            gamma_h = friction * dt
            c1_f = math.exp(-gamma_h)
            c1 = torch.tensor(c1_f, device=mass_diag.device)
            c2 = math.sqrt(1 - c1_f ** 2) * mass_diag.sqrt()
        return c1, c2

    def step(
        self,
        phi: torch.Tensor,              # [M, phi_dim]
        p: torch.Tensor,                # [M, phi_dim]
        grad_fn: Callable,              # phi → [M, phi_dim]
        friction,                       # float OR 0-dim tensor
        M_diag: torch.Tensor,           # [phi_dim]
        Gu: Optional[torch.Tensor] = None,   # [M, phi_dim] per-particle control (mode B)
        girsanov_log_weight: Optional[torch.Tensor] = None,  # [M] accumulated
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Single BAOAB step with optional per-particle control in the OU step.

        The controlled SDE is (mode B):
            dp = -∇U dt - Γ M^{-1} p dt + Gu_m dt + √(2Γ) dW

        Control Gu [M, phi_dim] is injected in the O step, not the B step, so the
        Girsanov correction (Archambeau et al. 2007) is exact.  In mode A, Gu=None
        and the step is standard BAOAB.

        B: p ← p - (h/2) ∇U(phi)
        A: phi ← phi + (h/2) M^{-1} p
        O: p ← c1*p + Gu*drift_coeff + c2*ξ   [+ Girsanov increment]
        A: phi ← phi + (h/2) M^{-1} p
        B: p ← p - (h/2) ∇U(phi)

        Returns (phi, p, updated_girsanov_log_weight).
        """
        h = self.dt
        M_inv = 1.0 / M_diag.clamp(min=1e-6)   # [phi_dim]
        c1, c2 = self._ou_coeffs(friction, h, M_diag)

        # ── B: first momentum half-kick (gradient only) ───────────────────────
        g = grad_fn(phi)                          # [M, phi_dim]
        p = p - (h / 2) * g

        # ── A: position half-step ─────────────────────────────────────────────
        phi = phi + (h / 2) * (M_inv.unsqueeze(0) * p)

        # ── O: Ornstein-Uhlenbeck + per-particle control injection ────────────
        # Control enters the OU step so the Girsanov correction is exact.
        # The mean shift from drift Gu over the OU interval is:
        #   ∫₀ʰ e^{-Γ(h-s)} Gu ds = (1 - e^{-Γh}) / Γ * Gu  ≈ h * Gu for small Γh
        noise = torch.randn_like(p)               # [M, phi_dim]

        if torch.is_tensor(friction) and friction.dim() == 0:
            gamma_h = friction * h
            drift_coeff = (1.0 - c1) / friction.clamp(min=1e-8) if gamma_h.abs() > 1e-6 \
                          else torch.tensor(float(h), device=p.device)
        else:
            gamma_h = float(friction) * h
            drift_coeff = (1.0 - float(c1)) / max(float(friction), 1e-8) \
                          if gamma_h > 1e-6 else h

        p = c1 * p + c2.unsqueeze(0) * noise
        if Gu is not None:
            p = p + drift_coeff * Gu              # [M, phi_dim] per-particle drift

        # ── Girsanov log-weight increment (mode B only) ───────────────────────
        # For the controlled OU  dp = -Γ M^{-1} p dt + Gu dt + √(2Γ) dW,
        # the per-particle Radon-Nikodym derivative is:
        #   δ log w_m = drift_coeff * (M_inv * Gu_m) · noise_m
        #               - (drift_coeff² / 2) * ||M^{-1/2} Gu_m||²
        # Both terms are per-particle (differ across m due to different Gu_m).
        if Gu is not None and girsanov_log_weight is not None:
            M_inv_Gu = M_inv.unsqueeze(0) * Gu                       # [M, phi_dim]
            stochastic     = drift_coeff * (M_inv_Gu * noise).sum(-1)  # [M]
            ito_correction = 0.5 * (drift_coeff ** 2) * (M_inv.unsqueeze(0) * Gu ** 2).sum(-1)  # [M]
            girsanov_log_weight = girsanov_log_weight + stochastic - ito_correction

        # ── A: position half-step ─────────────────────────────────────────────
        phi = phi + (h / 2) * (M_inv.unsqueeze(0) * p)

        # ── B: second momentum half-kick (gradient only) ──────────────────────
        g = grad_fn(phi)
        p = p - (h / 2) * g

        return phi, p, girsanov_log_weight

    def integrate(
        self,
        phi0: torch.Tensor,       # [M, phi_dim]
        grad_fn: Callable,        # phi → [M, phi_dim]  (force, beta=1 always)
        n_steps: int,
        friction,                 # float OR 0-dim tensor (differentiable)
        M_diag: torch.Tensor,     # [phi_dim]  (differentiable)
        Gu: Optional[torch.Tensor] = None,   # [M, phi_dim] per-particle control
        accumulate_girsanov: bool = False,
        dt_scale = 1.0,           # float OR 0-dim tensor — timestep multiplier
        # Legacy G/u params kept for API compat but ignored (use Gu instead)
        G: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate BAOAB for n_steps with optional per-particle control Gu.

        Gu [M, phi_dim] is the precomputed per-particle control vector from
        compute_per_particle_Gu(). If None, runs standard mode A BAOAB.

        dt_scale multiplies the effective timestep: h_eff = dt * dt_scale.
        Both drift and OU noise scale with h_eff so the stationary distribution
        exp(-U(phi)) is unchanged — only mixing speed varies.

        Returns dict with:
            'phi'            — [M, phi_dim] final particles
            'p'              — [M, phi_dim] final momenta
            'girsanov_log_w' — [M] accumulated Girsanov correction (zeros if mode A)
        """
        M = phi0.shape[0]
        p0 = torch.randn_like(phi0) * M_diag.unsqueeze(0).sqrt()

        # Girsanov accumulator (mode B only)
        g_log_w = torch.zeros(M, device=phi0.device) if (accumulate_girsanov and Gu is not None) \
                  else None

        # Apply dt_scale: temporarily override self.dt
        orig_dt = self.dt
        if torch.is_tensor(dt_scale):
            self.dt = orig_dt * dt_scale
        else:
            self.dt = orig_dt * float(dt_scale)

        try:
            phi, p = phi0, p0
            for _ in range(n_steps):
                phi, p, g_log_w = self.step(phi, p, grad_fn, friction, M_diag, Gu, g_log_w)
        finally:
            self.dt = orig_dt   # always restore, even on exception

        return {
            "phi": phi,
            "p": p,
            "girsanov_log_w": g_log_w if g_log_w is not None else torch.zeros(M, device=phi0.device),
        }
        h = self.dt
        M_inv = 1.0 / M_diag.clamp(min=1e-6)          # [phi_dim]  differentiable
        c1, c2 = self._ou_coeffs(friction, h, M_diag)

        # Control signal Gu [phi_dim] — injected into the OU diffusion step (mode B).
        # Placing Gu *inside* the O step (not in B) means the controlled SDE is:
        #   dp = -∇U dt - Γ M^{-1} p dt + Gu dt + √(2Γ) dW
        # which matches the Archambeau et al. 2007 formulation exactly.
        # The Girsanov correction is then a proper stochastic integral and
        # correctly restores SMC unbiasedness.
        # In mode A (Gu=None) this block is skipped entirely.
        Gu = (G @ u) if (G is not None and u is not None) else None

        # ── B: first momentum half-kick (gradient only, no control) ──────────
        g = grad_fn(phi)                               # [M, phi_dim]
        p = p - (h / 2) * g

        # ── A: position half-step ─────────────────────────────────────────────
        phi = phi + (h / 2) * (M_inv.unsqueeze(0) * p)

        # ── O: Ornstein-Uhlenbeck + control injection ─────────────────────────
        # Control enters here so the Girsanov correction is exact.
        # The OU-with-drift SDE  dp = -Γ M^{-1} p dt + Gu dt + √(2Γ) dW
        # discretises as:  p_new = c1*p + (1-c1)/Γ * Gu + c2*ξ
        # For small h: (1-c1)/Γ ≈ h, so the drift contribution is ≈ h*Gu.
        noise = torch.randn_like(p)
        p_before_ou = p

        # Exact mean-shift from drift Gu over the OU interval
        # (1 - e^{-Γh}) / Γ  ≈  h for small Γh, exact otherwise
        if torch.is_tensor(friction) and friction.dim() == 0:
            gamma_h = friction * h
            if torch.abs(gamma_h) > 1e-6:
                drift_coeff = (1.0 - c1) / friction
            else:
                drift_coeff = torch.tensor(float(h), device=p.device)
        else:
            gamma_h = float(friction) * h
            drift_coeff = (1.0 - float(c1)) / max(float(friction), 1e-8) if gamma_h > 1e-6 else h

        p = c1 * p + c2.unsqueeze(0) * noise
        if Gu is not None:
            p = p + drift_coeff * Gu.unsqueeze(0)     # mean shift from controlled drift

        # ── Girsanov log-weight increment (mode B only) ───────────────────────
        # For the controlled OU  dp = -Γ M^{-1} p dt + Gu dt + √(2Γ) dW,
        # the Radon-Nikodym derivative w.r.t. the uncontrolled OU is:
        #   log(dP_ctrl / dP_free) = (1/2Γ) ∫ Gu · dW  -  (1/4Γ) ∫ ||Gu||² dt
        # Discretely (Euler-Maruyama of the stochastic integral):
        #   δ log w = (drift_coeff / c2²) * (Gu · noise * c2)
        #             - (drift_coeff² / (2*c2²)) * ||Gu||²
        #   simplifies to:
        #   δ log w = drift_coeff * (M_inv_Gu · noise)
        #             - (drift_coeff² / 2) * (M_inv * Gu²).sum()
        # where M_inv_Gu absorbs the 1/c2² factor via the noise variance.
        if Gu is not None and girsanov_log_weight is not None:
            M_inv_Gu = M_inv * Gu                                       # [phi_dim]
            # Stochastic integral estimate: Gu · dW, where dW = noise (unit variance)
            stochastic  = drift_coeff * (M_inv_Gu.unsqueeze(0) * noise).sum(-1)  # [M]
            # Itô correction: -(drift_coeff²/2) * ||M^{-1/2} Gu||²
            ito_correction = 0.5 * (drift_coeff ** 2) * (M_inv * Gu ** 2).sum()
            girsanov_log_weight = girsanov_log_weight + stochastic - ito_correction

        # ── A: position half-step ─────────────────────────────────────────────
        phi = phi + (h / 2) * (M_inv.unsqueeze(0) * p)

        # ── B: second momentum half-kick (gradient only, no control) ─────────
        g = grad_fn(phi)
        p = p - (h / 2) * g

        return phi, p, girsanov_log_weight

    def integrate(
        self,
        phi0: torch.Tensor,
        grad_fn: Callable,
        n_steps: int,
        friction,
        M_diag: torch.Tensor,
        Gu: Optional[torch.Tensor] = None,   # ← ADD: [M, phi_dim] per-particle control
        accumulate_girsanov: bool = False,
        dt_scale = 1.0,
        # Legacy — kept for compat, ignored (use Gu instead)
        G: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate BAOAB for n_steps with optional timestep scaling.

        dt_scale multiplies the effective timestep: h_eff = dt * dt_scale.
        Both drift and OU noise scale with h_eff, so the stationary
        distribution exp(-U(phi)) is UNCHANGED — only mixing speed varies.
        This makes dt_scale a target-neutral exploration parameter, unlike
        beta (energy scaler) which moves the stationary distribution.

        grad_fn returns the true unscaled force (beta=1 assumed):
            grad_fn(phi) = nabla_U(phi; X)

        Returns dict with:
            'phi'   — [M, phi_dim] final particles
            'p'     — [M, phi_dim] final momenta
            'girsanov_log_w'  — [M] accumulated Girsanov correction (0 if mode A)
        """
        M = phi0.shape[0]
        p0 = torch.randn_like(phi0) * M_diag.unsqueeze(0).sqrt()

        # Girsanov accumulator (only used in mode B)
        if accumulate_girsanov and Gu is not None:
            g_log_w = torch.zeros(M, device=phi0.device)
        else:
            g_log_w = None

        # Apply dt_scale for this integrate call: temporarily override self.dt
        orig_dt = self.dt
        if torch.is_tensor(dt_scale):
            self.dt = orig_dt * dt_scale        # stays in graph if dt_scale is tensor
        else:
            self.dt = orig_dt * float(dt_scale)

        try:
            phi, p = phi0, p0
            for _ in range(n_steps):
                phi, p, g_log_w = self.step(phi, p, grad_fn, friction, M_diag, Gu, g_log_w)
        finally:
            self.dt = orig_dt   # always restore, even if a step raises

        return {
            "phi": phi,
            "p": p,
            "girsanov_log_w": g_log_w if g_log_w is not None else torch.zeros(M, device=phi0.device),
        }


# ── SMC weight update and resampling ─────────────────────────────────────────

def smc_weight_update(
    log_weights: torch.Tensor,          # [M] current log-weights
    phi: torch.Tensor,                  # [M, phi_dim] particles
    X_k: torch.Tensor,                  # [N, D] current batch
    energy_fn,                          # GMMEnergy instance
    beta: float,                        # navigator beta — NOT used for weights
    girsanov_log_w: Optional[torch.Tensor] = None,  # [M] from integrate()
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SMC incremental weight update targeting the TRUE posterior (beta_w = 1).

    Weight update always uses beta_w = 1 regardless of the navigator output:

        log w_k = log w_{k-1} + log p(X_k | phi)

    so that {phi_m, w_m} consistently approximates p(phi | X_{1:k}).

    Why beta_w = 1 even though BAOAB uses a navigator-tuned beta:
        The navigator's beta controls the *proposal* (BAOAB energy landscape)
        and is a learned geometric parameter.  Using it in the weight update
        would make weights represent prod_j p(X_j|phi)^{beta_j} — a different
        target each episode.  Separating the two means:
            - BAOAB beta: shapes the proposal (how particles move)
            - weight beta: always 1 (always tracking the same posterior)
        This is the standard generalized-Bayes / power-likelihood distinction.

    For mode B (controlled dynamics), adds Girsanov correction to restore
    unbiasedness under the biased (controlled) proposal:
        log w_k = log w_{k-1} + log p(X_k|phi) + girsanov_log_w

    The `beta` argument is accepted for API compatibility but is not used.

    Returns:
        log_weights: [M] updated unnormalized
        weights:     [M] normalized
    """
    with torch.no_grad():
        # log p(X_k | phi) per particle — no prior (prior enters via BAOAB force)
        log_lik = -energy_fn.log_likelihood(phi, X_k)   # [M], log_likelihood returns -logp
        log_inc = log_lik                                # = +log p(X_k | phi)

        if girsanov_log_w is not None:
            log_inc = log_inc + girsanov_log_w.detach()

    log_w_new = log_weights + log_inc
    log_w_new = log_w_new - log_w_new.max()   # numerical stability
    w_norm = torch.softmax(log_w_new, dim=0)
    return log_w_new, w_norm


def compute_ess(weights: torch.Tensor) -> float:
    """ESS = (sum w)^2 / sum w^2  for normalized weights."""
    return float((weights.sum() ** 2 / (weights ** 2).sum().clamp(min=1e-30)).item())


def systematic_resample(
    phi: torch.Tensor,           # [M, phi_dim]
    weights: torch.Tensor,       # [M] normalized
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Systematic resampling (low-variance).

    Returns:
        phi_resampled:   [M, phi_dim]
        weights_uniform: [M] uniform 1/M
    """
    M = phi.shape[0]
    cumsum = weights.cumsum(dim=0)

    # Single random offset
    u0 = torch.rand(1, device=device) / M
    positions = u0 + torch.arange(M, device=device).float() / M

    # Indices
    indices = torch.searchsorted(cumsum, positions).clamp(0, M - 1)
    phi_new = phi[indices]
    w_new = torch.ones(M, device=device) / M

    return phi_new, w_new