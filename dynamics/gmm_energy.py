"""
GMM Free Energy Functional.

Defines U(phi; X) = -ELBO for a Gaussian Mixture Model.
This is the potential energy landscape that BAOAB particles descend.

ELBO = E_q[log p(X, z)] - E_q[log q(z)]
     = sum_n log(sum_k pi_k N(x_n; mu_k, Sigma_k))  [for hard/soft EM]

We use the marginal log-likelihood directly (no explicit responsibilities)
since we're optimizing over GMM parameters, not a VAE encoder.

Free energy:
    U(phi; X) = -sum_{n=1}^{N} log p(x_n | phi)
    where p(x_n | phi) = sum_k pi_k N(x_n; mu_k, L_k L_k^T)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.gmm_problem import unpack_phi, L_vec_to_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Core log-likelihood
# ──────────────────────────────────────────────────────────────────────────────

def gmm_log_likelihood(
    phi: torch.Tensor,   # [M, phi_dim]
    X: torch.Tensor,     # [N, D]
    K: int,
    D: int,
) -> torch.Tensor:
    """
    Compute per-particle log p(X | phi) = sum_n log sum_k pi_k N(x_n; mu_k, Sigma_k).

    Args:
        phi: [M, phi_dim] particles
        X:   [N, D] data batch
        K:   number of components
        D:   data dimension
    Returns:
        log_lik: [M] per-particle total log-likelihood
    """
    M = phi.shape[0]
    N = X.shape[0]

    pi_tilde, mu, L_vecs = unpack_phi(phi, K, D)  # [M,K], [M,K,D], [M,K,chol]

    # Mixing weights
    log_pi = F.log_softmax(pi_tilde, dim=-1)  # [M, K]

    # Build Cholesky factors
    L = L_vec_to_matrix(L_vecs, D)  # [M, K, D, D]

    # Compute log N(x_n; mu_k, L L^T) for all n, k, m
    # x_n - mu_k : [M, K, N, D]  via broadcasting
    # X:  [N, D]   →  [1, 1, N, D]
    # mu: [M, K, D] → [M, K, 1, D]
    diff = X[None, None, :, :] - mu[:, :, None, :]  # [M, K, N, D]

    # Solve L @ alpha = diff  (per component, per particle)
    # L: [M, K, D, D]  diff: [M, K, N, D]
    # We need alpha = L^{-1} diff for Mahalanobis distance

    # Reshape for batch triangular solve
    MK = M * K
    L_flat = L.reshape(MK, D, D)                      # [M*K, D, D]
    diff_flat = diff.reshape(MK, N, D).permute(0, 2, 1)  # [M*K, D, N]

    # Solve: L @ alpha = diff  →  alpha = L^{-1} diff
    alpha = torch.linalg.solve_triangular(L_flat, diff_flat, upper=False)  # [M*K, D, N]

    # Mahalanobis squared: ||alpha||^2 per (m, k, n)
    maha = (alpha ** 2).sum(dim=1)  # [M*K, N]
    maha = maha.reshape(M, K, N)   # [M, K, N]

    # Log determinant: log|Sigma| = 2 * sum_d log|L[d,d]|
    log_diag = torch.log(L_flat.diagonal(dim1=-2, dim2=-1).abs().clamp(min=1e-8))
    log_det = 2.0 * log_diag.sum(dim=-1)  # [M*K]
    log_det = log_det.reshape(M, K)        # [M, K]

    # Log Gaussian density: [M, K, N]
    log_norm_const = -0.5 * (D * math.log(2 * math.pi) + log_det[:, :, None])
    log_gauss = log_norm_const - 0.5 * maha  # [M, K, N]

    # Log mixture: log sum_k exp(log_pi_k + log_gauss_k)  →  [M, N]
    log_mix = torch.logsumexp(
        log_pi[:, :, None] + log_gauss,  # [M, K, N]
        dim=1,
    )  # [M, N]

    return log_mix.sum(dim=-1)  # [M]  sum over data points


# ──────────────────────────────────────────────────────────────────────────────
# Free energy class
# ──────────────────────────────────────────────────────────────────────────────

class GMMEnergy(nn.Module):
    """
    GMM Free Energy: U(phi; X) = -log p(X | phi) + lambda_prior * ||phi||^2 / 2.

    This is the potential for BAOAB transport.
    Gradients ∇_phi U are computed via autograd.
    """

    def __init__(
        self,
        K: int,
        D: int,
        lambda_prior: float = 0.01,
    ):
        super().__init__()
        self.K = K
        self.D = D
        self.lambda_prior = lambda_prior

    def free_energy(
        self,
        phi: torch.Tensor,   # [M, phi_dim]
        X: torch.Tensor,     # [N, D]
    ) -> torch.Tensor:
        """
        U(phi; X) = -log p(X | phi) + lambda_prior * ||phi||^2 / 2

        Used for BAOAB gradient (force field targets the full posterior).
        Prior is applied through the gradient, not through SMC weights.
        """
        log_lik = gmm_log_likelihood(phi, X, self.K, self.D)   # [M]
        prior = 0.5 * (phi ** 2).sum(dim=-1)                   # [M]
        return -log_lik + self.lambda_prior * prior

    def log_likelihood(
        self,
        phi: torch.Tensor,   # [M, phi_dim]
        X: torch.Tensor,     # [N, D]
    ) -> torch.Tensor:
        """
        Returns -log p(X | phi) per particle (no prior term).

        Used for SMC incremental weight updates only:
            log w_k += -beta * log_likelihood(phi; X_k)

        Using free_energy here would apply the prior T times across
        T streaming batches, making the effective target:
            p(phi | X)^T * p(phi)^T  instead of  p(phi | X) * p(phi).
        The prior is incorporated correctly once via the BAOAB force.
        """
        return -gmm_log_likelihood(phi, X, self.K, self.D)     # [M]

    def gradient(
        self,
        phi: torch.Tensor,   # [M, phi_dim]
        X: torch.Tensor,     # [N, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ∇_phi U(phi; X) via autograd.

        Returns:
            grad: [M, phi_dim]
            energy: [M]
        """
        phi_req = phi.detach().requires_grad_(True)

        with torch.enable_grad():
            U = self.free_energy(phi_req, X)       # [M]
            grad = torch.autograd.grad(
                U.sum(), phi_req,
                create_graph=False,
                retain_graph=False,
            )[0]                                   # [M, phi_dim]

        return grad, U.detach()

    def make_grad_fn(
        self,
        X: torch.Tensor,
        create_graph: bool = False,
    ):
        """
        Returns a closure  grad_fn(phi) → nabla_U(phi; X).

        beta is NOT a parameter here — the force always uses beta=1 so that
        BAOAB targets exp(-U(phi)), the true posterior.  Mixing speed is
        controlled by dt_scale in the integrator, which is target-neutral.

        create_graph (bool):
            False (outside gradient window):
                phi is detached — first-order BAOAB. M_diag and gamma
                receive gradients via the position/momentum update equations,
                not through second-order force terms. Memory-efficient.
            True (inside last n_grad_batches):
                phi participates in the graph — second-order path
                loss → phi_final → g(phi_k) → phi_k enabled.
                Needed for dt_scale to receive gradients through force.
        """
        def grad_fn(phi: torch.Tensor) -> torch.Tensor:
            if create_graph:
                phi_g = phi if phi.requires_grad else phi.detach().requires_grad_(True)
                with torch.enable_grad():
                    U = self.free_energy(phi_g, X)
                    g = torch.autograd.grad(
                        U.sum(), phi_g,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                return g   # no beta multiplier — beta=1 always
            else:
                phi_req = phi.detach().requires_grad_(True)
                with torch.enable_grad():
                    U = self.free_energy(phi_req, X)
                    g = torch.autograd.grad(U.sum(), phi_req)[0]
                return g.detach()   # first-order; g treated as constant

        return grad_fn

    def sense(
        self,
        phi_mean: torch.Tensor,   # [phi_dim] mean particle
        X_k: torch.Tensor,         # [B, D] current batch
        X_prev: Optional[torch.Tensor] = None,  # previous batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sense features: sufficient-stat residual nu and delta_F.

        nu = sufficient_stats(X_k) - predicted_stats(phi_mean)
        dF = F(phi_mean; X_k) - F(phi_mean; X_prev)   (if X_prev provided)

        Args:
            phi_mean: [phi_dim] mean of current particles
            X_k:   [B, D] current batch
            X_prev: [B', D] previous batch (for dF)
        Returns:
            nu:  [feat_dim_nu] sufficient stat residual
            dF:  scalar tensor
        """
        pi_tilde, mu, L_vecs = unpack_phi(phi_mean.unsqueeze(0), self.K, self.D)
        pi = torch.softmax(pi_tilde.squeeze(0), dim=-1)  # [K]
        mu_m = mu.squeeze(0)                              # [K, D]
        L = L_vec_to_matrix(L_vecs.squeeze(0), self.D)  # [K, D, D]
        Sigma = torch.bmm(L, L.transpose(-1, -2))         # [K, D, D]

        # Empirical sufficient stats from X_k
        N_k = X_k.shape[0]
        # Soft responsibilities r_{nk} = pi_k N(x_n; mu_k, Sigma_k) / Z_n
        r = _compute_responsibilities(X_k, pi, mu_m, L, self.K, self.D)  # [N, K]

        N_hat = r.sum(dim=0)                           # [K] predicted counts
        N_emp = torch.ones(self.K, device=X_k.device) * (N_k / self.K)  # uniform baseline
        sum_x_hat = (r.unsqueeze(-1) * X_k.unsqueeze(1)).sum(0)  # [K, D]
        sum_x_emp = X_k.mean(0, keepdim=True).expand(self.K, -1)  # [K, D]

        # nu = concat of (N_hat - N_emp, sum_x_hat - sum_x_emp flattened)
        nu = torch.cat([
            (N_hat - N_emp) / N_k,                    # [K]
            (sum_x_hat - sum_x_emp).flatten() / N_k,  # [K*D]
        ])  # [K + K*D]

        # dF = energy difference
        U_curr = self.free_energy(phi_mean.unsqueeze(0), X_k).squeeze()
        if X_prev is not None and X_prev.shape[0] > 0:
            U_prev = self.free_energy(phi_mean.unsqueeze(0), X_prev).squeeze()
            dF = U_curr - U_prev
        else:
            dF = torch.zeros(1, device=X_k.device).squeeze()

        return nu.detach(), dF.detach()

    def importance_weights(
        self,
        phi: torch.Tensor,   # [M, phi_dim]
        X: torch.Tensor,     # [N, D]
        beta: float = 1.0,
        log_weights_prev: Optional[torch.Tensor] = None,  # [M]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SMC importance weights for current batch.

        w_k^(m) ∝ w_{k-1}^(m) * exp(-beta * ell_k(phi^(m)))
        where ell_k = free_energy on current batch

        Returns:
            log_weights: [M] unnormalized log weights
            weights:     [M] normalized weights
        """
        with torch.no_grad():
            U = self.free_energy(phi, X)   # [M]
            log_inc = -beta * U

            if log_weights_prev is not None:
                log_w = log_weights_prev + log_inc
            else:
                log_w = log_inc

        # Normalize
        log_w = log_w - torch.logsumexp(log_w, dim=0)
        w = torch.exp(log_w)
        return log_w, w


def _compute_responsibilities(
    X: torch.Tensor,        # [N, D]
    pi: torch.Tensor,       # [K]
    mu: torch.Tensor,       # [K, D]
    L: torch.Tensor,        # [K, D, D]
    K: int,
    D: int,
) -> torch.Tensor:
    """
    Compute soft responsibilities r_{nk} ∝ pi_k N(x_n; mu_k, L_k L_k^T).

    Returns: [N, K]
    """
    N = X.shape[0]

    log_pi = torch.log(pi.clamp(min=1e-8))  # [K]

    # Mahalanobis distances
    diff = X[:, None, :] - mu[None, :, :]  # [N, K, D]

    # L^{-1} diff: [N, K, D]
    diff_T = diff.permute(1, 2, 0)   # [K, D, N]
    alpha = torch.linalg.solve_triangular(L, diff_T, upper=False)  # [K, D, N]
    alpha = alpha.permute(2, 0, 1)   # [N, K, D]

    maha = (alpha ** 2).sum(dim=-1)  # [N, K]

    log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).abs().clamp(min=1e-8).log().sum(dim=-1)  # [K]
    log_norm = -0.5 * (D * math.log(2 * math.pi) + log_det[None, :])  # [1, K]

    log_gauss = log_norm - 0.5 * maha  # [N, K]
    log_resp = log_pi[None, :] + log_gauss  # [N, K]

    # Softmax to normalize
    r = torch.softmax(log_resp, dim=-1)  # [N, K]
    return r
