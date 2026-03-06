"""
Canonicalization of GMM particles.

GMM has K! permutation symmetry. We canonicalize by sorting components
by first coordinate of their mean mu_k[0]. This must be applied
consistently to ALL coupled fields (pi, mu, L) simultaneously.

This prevents symmetry-related duplicate modes in PHC clustering.
"""
from __future__ import annotations

import torch

from data.gmm_problem import unpack_phi, pack_phi


def canonicalize_phi(phi: torch.Tensor, K: int, D: int) -> torch.Tensor:
    """
    Canonicalize particle state(s) by sorting components on mu[k, 0].

    Works for single particles [phi_dim] or batches [M, phi_dim].

    Args:
        phi:  [phi_dim] or [M, phi_dim]
        K:    number of GMM components
        D:    data dimension
    Returns:
        phi_canon: same shape as input, with components sorted
    """
    single = phi.ndim == 1
    if single:
        phi = phi.unsqueeze(0)

    pi_tilde, mu, L_vecs = unpack_phi(phi, K, D)  # [M,K], [M,K,D], [M,K,chol]

    # Sort by first coordinate of each mean
    sort_key = mu[..., 0]          # [M, K]
    perm = torch.argsort(sort_key, dim=-1)  # [M, K]

    # Apply permutation to all fields
    M = phi.shape[0]
    pi_tilde_c = _permute_batch(pi_tilde, perm)   # [M, K]
    mu_c       = _permute_batch(mu,       perm)   # [M, K, D]
    L_vecs_c   = _permute_batch(L_vecs,   perm)   # [M, K, chol]

    phi_canon = pack_phi_batch(pi_tilde_c, mu_c, L_vecs_c)  # [M, phi_dim]

    if single:
        phi_canon = phi_canon.squeeze(0)

    return phi_canon


def _permute_batch(tensor: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    Apply per-sample permutation along the K dimension.

    Args:
        tensor: [M, K, ...] arbitrary trailing dims
        perm:   [M, K] permutation indices
    Returns:
        permuted: [M, K, ...] reordered
    """
    M, K = perm.shape
    trailing = tensor.shape[2:]

    # Expand perm to match trailing dims
    perm_exp = perm.view(M, K, *([1] * len(trailing)))
    perm_exp = perm_exp.expand_as(tensor)

    return torch.gather(tensor, 1, perm_exp)


def pack_phi_batch(
    pi_tilde: torch.Tensor,  # [M, K]
    mu: torch.Tensor,         # [M, K, D]
    L_vecs: torch.Tensor,     # [M, K, chol_sz]
) -> torch.Tensor:
    """Pack batch of (pi_tilde, mu, L_vecs) into [M, phi_dim]."""
    M = pi_tilde.shape[0]
    return torch.cat([pi_tilde, mu.reshape(M, -1), L_vecs.reshape(M, -1)], dim=-1)


def hungarian_match(
    mu_pred: torch.Tensor,  # [K, D] or [M, K, D] predicted means
    mu_true: torch.Tensor,  # [K, D] true means
) -> torch.Tensor:
    """
    Find best assignment of predicted clusters to true clusters via Hungarian.

    For single prediction [K, D]:
        Returns permutation perm such that mu_pred[perm[k]] ≈ mu_true[k]

    Args:
        mu_pred: [K, D] or [M, K, D]
        mu_true: [K, D]
    Returns:
        perm: [K] or [M, K] assignment indices
    """
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
    except ImportError:
        raise ImportError("scipy required for Hungarian matching: pip install scipy")

    single = mu_pred.ndim == 2
    if single:
        mu_pred = mu_pred.unsqueeze(0)

    M, K, D = mu_pred.shape
    perms = []

    mu_true_np = mu_true.detach().cpu().numpy()  # [K, D]

    for m in range(M):
        mu_p_np = mu_pred[m].detach().cpu().numpy()  # [K, D]

        # Cost matrix: pairwise L2 distances [K, K]
        cost = np.sum((mu_p_np[:, None, :] - mu_true_np[None, :, :]) ** 2, axis=-1)

        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.zeros(K, dtype=np.int64)
        perm[col_ind] = row_ind
        perms.append(torch.tensor(perm, dtype=torch.long, device=mu_pred.device))

    perm_tensor = torch.stack(perms)  # [M, K]

    if single:
        perm_tensor = perm_tensor.squeeze(0)

    return perm_tensor
