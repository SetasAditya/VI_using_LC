"""
Weighted Kernel Density Estimation for PHC.

Builds the clustering Hamiltonian H^clust(z) = -log(rho(z) + eps)
from weighted particle embeddings.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch


def silverman_bandwidth(z: torch.Tensor) -> float:
    """
    Silverman's rule-of-thumb bandwidth for Gaussian KDE.

    sigma = n^{-1/(d+4)} * std(z)  (averaged over dims)

    Args:
        z: [M, d] particle embeddings
    Returns:
        sigma: scalar bandwidth
    """
    M, d = z.shape
    std = z.std(dim=0).mean().item()
    sigma = M ** (-1.0 / (d + 4.0)) * std
    return max(sigma, 1e-3)


def scott_bandwidth(z: torch.Tensor) -> float:
    """
    Scott's rule bandwidth.

    sigma = n^{-1/(d+4)} * n^{1/d} * std(z)   [simplified]
    """
    M, d = z.shape
    std = z.std(dim=0).mean().item()
    sigma = M ** (-1.0 / (d + 4.0)) * std * M ** (1.0 / d)
    return max(sigma, 1e-3)


def weighted_kde(
    z: torch.Tensor,              # [M, d] particle embeddings
    weights: torch.Tensor,         # [M] normalized importance weights
    eval_points: torch.Tensor,     # [P, d] evaluation locations
    sigma: Optional[float] = None, # bandwidth (auto if None)
    M_metric: Optional[torch.Tensor] = None,  # [d, d] or [d] metric tensor
    sigma_rule: str = "silverman",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Evaluate weighted Gaussian KDE at eval_points.

    rho(z) = sum_m w_m K_sigma(d_M(z, z_m))
    where K_sigma(r) = exp(-r^2 / (2 sigma^2)) / Z

    Args:
        z:           [M, d] particle embeddings
        weights:     [M] normalized weights (sum to 1)
        eval_points: [P, d] where to evaluate density
        sigma:       bandwidth (auto-computed if None)
        M_metric:    metric tensor for distances ([d,d] full or [d] diagonal)
        sigma_rule:  'silverman' | 'scott' | 'fixed'
        eps:         floor for density values
    Returns:
        rho: [P] density values
    """
    M, d = z.shape
    P = eval_points.shape[0]

    # Auto-compute bandwidth
    if sigma is None:
        if sigma_rule == "silverman":
            sigma = silverman_bandwidth(z)
        elif sigma_rule == "scott":
            sigma = scott_bandwidth(z)
        else:
            sigma = 1.0

    # Pairwise distances under metric M_k
    # diff: [P, M, d]
    diff = eval_points[:, None, :] - z[None, :, :]  # [P, M, d]

    if M_metric is not None:
        if M_metric.ndim == 1:
            # Diagonal metric: [d]
            dist_sq = (diff ** 2 * M_metric[None, None, :]).sum(dim=-1)  # [P, M]
        else:
            # Full metric: [d, d]
            # d_{M}(a, b)^2 = (a-b)^T M (a-b)
            # diff: [P, M, d]  →  flatten to [P*M, d]
            PM = P * M
            diff_flat = diff.reshape(PM, d)
            dist_sq = (diff_flat @ M_metric * diff_flat).sum(dim=-1)  # [P*M]
            dist_sq = dist_sq.reshape(P, M)
    else:
        dist_sq = (diff ** 2).sum(dim=-1)  # [P, M] Euclidean

    # Gaussian kernel
    log_kernel = -0.5 * dist_sq / (sigma ** 2)  # [P, M]

    # Normalize kernel (optional, affects absolute scale but not clustering)
    log_norm = -0.5 * d * math.log(2 * math.pi * sigma ** 2)
    log_kernel = log_kernel + log_norm

    # Weighted sum: rho(z) = sum_m w_m k(z, z_m)
    # In log space: log rho = logsumexp(log(w_m) + log_kernel_m)
    log_w = torch.log(weights.clamp(min=1e-30))  # [M]
    log_rho = torch.logsumexp(log_w[None, :] + log_kernel, dim=-1)  # [P]

    rho = torch.exp(log_rho)
    return rho.clamp(min=eps)


def cluster_hamiltonian(
    z: torch.Tensor,              # [M, d] particle embeddings
    weights: torch.Tensor,         # [M] normalized weights
    sigma: Optional[float] = None,
    M_metric: Optional[torch.Tensor] = None,
    sigma_rule: str = "silverman",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute H^clust(z_m) = -log(rho(z_m) + eps) at each particle.

    Returns: [M] Hamiltonian values, one per particle
    """
    rho = weighted_kde(
        z=z,
        weights=weights,
        eval_points=z,
        sigma=sigma,
        M_metric=M_metric,
        sigma_rule=sigma_rule,
        eps=eps,
    )
    return -torch.log(rho)  # [M]
