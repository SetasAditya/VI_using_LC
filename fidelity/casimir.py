"""
Casimir / Structural Invariant Checker.

Checks structural constraints that should be preserved by valid GMM particles:
    1. Simplex: |1 - sum(softmax(pi_tilde))| ≈ 0 (always 0 by construction)
    2. SPD: min eigenvalue of Sigma_k = L_k L_k^T > 0
    3. Energy dissipation: F_new <= F_old + tolerance (transport should descend)
    4. Cycle consistency: ||phi - reconstruct(embed(phi))||  (if embedder invertible)

These are independent of learned components — purely structural checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from data.gmm_problem import unpack_phi, L_vec_to_matrix


@dataclass
class CasimirResult:
    """Result of structural invariant checks."""
    eps_pi: float          # simplex violation (should be ~0)
    eps_Sigma: float       # min negative eigenvalue (should be 0)
    delta_F: float         # energy change this step (should be ≤ 0)
    flag: bool             # True if any violation above threshold

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert to feature tensor for loss computation."""
        return torch.tensor(
            [self.eps_pi, self.eps_Sigma, max(0.0, self.delta_F)],
            device=device, dtype=torch.float32,
        )


class CasimirChecker:
    """
    Checks structural invariants of GMM particles.

    Used as:
        1. Diagnostic during inference (flag violations)
        2. Loss term during training (penalize violations)
    """

    def __init__(
        self,
        K: int,
        D: int,
        tau_simplex: float = 1e-4,
        tau_spd: float = 1e-6,
        tau_energy: float = 0.1,
    ):
        self.K = K
        self.D = D
        self.tau_simplex = tau_simplex
        self.tau_spd = tau_spd
        self.tau_energy = tau_energy

    def check(
        self,
        phi: torch.Tensor,          # [M, phi_dim]
        F_old: Optional[float] = None,
        F_new: Optional[float] = None,
    ) -> CasimirResult:
        """
        Check structural invariants on particle batch.

        Computes mean violation across particles.

        Args:
            phi:    [M, phi_dim] particles
            F_old:  previous mean free energy (for delta_F check)
            F_new:  current mean free energy
        Returns:
            CasimirResult
        """
        M, phi_dim = phi.shape

        pi_tilde, mu, L_vecs = unpack_phi(phi, self.K, self.D)

        # 1. Simplex check: softmax always sums to 1, so eps_pi ≈ 0
        pi = torch.softmax(pi_tilde, dim=-1)           # [M, K]
        eps_pi = (pi.sum(dim=-1) - 1.0).abs().mean().item()

        # 2. SPD check: all Cholesky diagonal entries must be > 0
        L = L_vec_to_matrix(L_vecs, self.D)            # [M, K, D, D]
        diag = L.diagonal(dim1=-2, dim2=-1)             # [M, K, D]
        min_diag = diag.min().item()
        eps_Sigma = max(0.0, -min_diag + self.tau_spd)

        # 3. Energy dissipation check
        if F_old is not None and F_new is not None:
            delta_F = F_new - F_old
        else:
            delta_F = 0.0

        flag = (
            eps_pi > self.tau_simplex or
            eps_Sigma > self.tau_spd or
            delta_F > self.tau_energy
        )

        return CasimirResult(
            eps_pi=eps_pi,
            eps_Sigma=eps_Sigma,
            delta_F=delta_F,
            flag=flag,
        )

    def casimir_loss(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Differentiable loss penalizing structural violations.

        Used during training to regularize the navigator.

        Returns: scalar loss
        """
        pi_tilde, mu, L_vecs = unpack_phi(phi, self.K, self.D)

        # SPD penalty: penalize small/negative diagonal entries
        L = L_vec_to_matrix(L_vecs, self.D)
        diag = L.diagonal(dim1=-2, dim2=-1)  # [M, K, D]
        spd_violation = torch.relu(-diag + 1e-3).pow(2).mean()

        # Mild pi concentration penalty (encourage non-degenerate mixtures)
        pi = torch.softmax(pi_tilde, dim=-1)  # [M, K]
        # Penalize very uneven mixing (entropy bonus)
        entropy = -(pi * torch.log(pi.clamp(min=1e-8))).sum(dim=-1)
        pi_penalty = torch.relu(math.log(self.K) * 0.1 - entropy).mean()

        return spd_violation + 0.1 * pi_penalty


import math
