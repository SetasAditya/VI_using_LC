from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.distributions as dist


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _chol_size(D: int) -> int:
    return D * (D + 1) // 2


def phi_dim(K: int, D: int) -> int:
    return K + K * D + K * _chol_size(D)


def pack_phi(pi_tilde: torch.Tensor, mu: torch.Tensor, L_vecs: torch.Tensor) -> torch.Tensor:
    return torch.cat([pi_tilde, mu.flatten(), L_vecs.flatten()])


def unpack_phi(phi: torch.Tensor, K: int, D: int):
    chol_sz = _chol_size(D)
    batch = phi.shape[:-1]
    pi_tilde = phi[..., :K]
    mu = phi[..., K:K + K * D].reshape(*batch, K, D)
    L_vecs = phi[..., K + K * D:].reshape(*batch, K, chol_sz)
    return pi_tilde, mu, L_vecs


def L_vec_to_matrix(L_vec: torch.Tensor, D: int) -> torch.Tensor:
    *batch, _ = L_vec.shape
    L = torch.zeros(*batch, D, D, dtype=L_vec.dtype, device=L_vec.device)
    idx = torch.tril_indices(D, D, device=L_vec.device)
    L[..., idx[0], idx[1]] = L_vec
    diag_idx = torch.arange(D, device=L_vec.device)
    L[..., diag_idx, diag_idx] = L[..., diag_idx, diag_idx].abs().clamp(min=1e-3)
    return L


def L_matrix_to_vec(L: torch.Tensor, D: int) -> torch.Tensor:
    idx = torch.tril_indices(D, D, device=L.device)
    return L[..., idx[0], idx[1]]


@dataclass
class GMMProblem:
    K: int
    D: int
    pi_true: torch.Tensor
    mu_true: torch.Tensor
    Sigma_true: torch.Tensor
    L_true: torch.Tensor
    X: torch.Tensor
    labels_true: torch.Tensor
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def to(self, device):
        self.pi_true = self.pi_true.to(device)
        self.mu_true = self.mu_true.to(device)
        self.Sigma_true = self.Sigma_true.to(device)
        self.L_true = self.L_true.to(device)
        self.X = self.X.to(device)
        self.labels_true = self.labels_true.to(device)
        self.device = device
        return self

    @property
    def N(self) -> int:
        return int(self.X.shape[0])

    @property
    def phi_dim(self) -> int:
        return phi_dim(self.K, self.D)

    def true_phi(self) -> torch.Tensor:
        pi_tilde = torch.log(self.pi_true + 1e-8)
        L_vecs = L_matrix_to_vec(self.L_true, self.D)
        return pack_phi(pi_tilde, self.mu_true, L_vecs)


# -----------------------------------------------------------------------------
# Problem sampler
# -----------------------------------------------------------------------------

def sample_gmm_problem(
    K: int,
    D: int,
    N: int,
    overlap: float = 0.3,
    sigma_scale: float = 1.0,
    device: torch.device = torch.device("cpu"),
    dirichlet_alpha: float = 2.0,
) -> GMMProblem:
    separation = sigma_scale / max(overlap, 0.05)

    alpha = torch.full((K,), float(dirichlet_alpha), device=device)
    pi_true = dist.Dirichlet(alpha).sample()
    pi_true = pi_true / pi_true.sum()

    mu_true = _sample_separated_means(K, D, separation, device)
    L_true = _sample_cholesky_factors(K, D, sigma_scale, device)
    Sigma_true = torch.bmm(L_true, L_true.transpose(-1, -2))
    X, labels_true = _sample_gmm_data(N, pi_true, mu_true, L_true, device)

    return GMMProblem(
        K=K,
        D=D,
        pi_true=pi_true,
        mu_true=mu_true,
        Sigma_true=Sigma_true,
        L_true=L_true,
        X=X,
        labels_true=labels_true,
        device=device,
    )


def _sample_separated_means(K: int, D: int, separation: float, device: torch.device) -> torch.Tensor:
    means = []
    max_attempts = 1000
    for _ in range(K):
        for _ in range(max_attempts):
            mu = torch.randn(D, device=device) * separation
            if not means or all(torch.norm(mu - m) >= separation * 0.5 for m in means):
                means.append(mu)
                break
        else:
            means.append(torch.randn(D, device=device) * separation)
    return torch.stack(means, dim=0)


def _sample_cholesky_factors(K: int, D: int, sigma_scale: float, device: torch.device) -> torch.Tensor:
    Ls = []
    for _ in range(K):
        L = torch.zeros(D, D, device=device)
        for i in range(D):
            for j in range(i):
                L[i, j] = torch.randn((), device=device) * sigma_scale * 0.3
            L[i, i] = torch.exp(torch.randn((), device=device) * 0.3) * sigma_scale
        Ls.append(L)
    return torch.stack(Ls, dim=0)


def _sample_gmm_data(N: int, pi: torch.Tensor, mu: torch.Tensor, L: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    K, D = mu.shape
    labels = dist.Categorical(pi).sample((N,))
    X = torch.zeros(N, D, device=device)
    for k in range(K):
        mask = labels == k
        n_k = int(mask.sum().item())
        if n_k == 0:
            continue
        eps = torch.randn(n_k, D, device=device)
        X[mask] = mu[k] + (L[k] @ eps.T).T
    return X, labels


# -----------------------------------------------------------------------------
# Streaming helpers
# -----------------------------------------------------------------------------

def get_streaming_batches(problem: GMMProblem, T_batches: int, batch_size: int, shuffle: bool = True) -> List[torch.Tensor]:
    X = problem.X
    if shuffle:
        perm = torch.randperm(problem.N, device=problem.device)
        X = X[perm]
    batches = []
    for t in range(T_batches):
        s = (t * batch_size) % problem.N
        e = min(s + batch_size, problem.N)
        batch = X[s:e]
        if batch.shape[0] < batch_size:
            batch = torch.cat([batch, X[: batch_size - batch.shape[0]]], dim=0)
        batches.append(batch)
    return batches


def get_streaming_index_batches(problem: GMMProblem, T_batches: int, batch_size: int, shuffle: bool = True) -> List[torch.Tensor]:
    idx = torch.arange(problem.N, device=problem.device)
    if shuffle:
        idx = idx[torch.randperm(problem.N, device=problem.device)]
    batches = []
    for t in range(T_batches):
        s = (t * batch_size) % problem.N
        e = min(s + batch_size, problem.N)
        batch = idx[s:e]
        if batch.shape[0] < batch_size:
            batch = torch.cat([batch, idx[: batch_size - batch.shape[0]]], dim=0)
        batches.append(batch)
    return batches


# -----------------------------------------------------------------------------
# Prior over phi (kept for future GMM-parameter transport experiments)
# -----------------------------------------------------------------------------

def _kmeans_pp_centers(X: torch.Tensor, K: int) -> torch.Tensor:
    N, D = X.shape
    device = X.device
    idx0 = torch.randint(N, (1,), device=device).item()
    centers = [X[idx0]]
    for _ in range(1, K):
        c_stack = torch.stack(centers, dim=0)
        diffs = X.unsqueeze(1) - c_stack.unsqueeze(0)
        dists = (diffs ** 2).sum(-1).min(dim=1).values.clamp(min=0.0)
        probs = dists / dists.sum().clamp(min=1e-12)
        idx = torch.multinomial(probs, 1).item()
        centers.append(X[idx])
    return torch.stack(centers, dim=0)


def sample_prior_particles(M: int, K: int, D: int, problem: Optional[GMMProblem] = None,
                           prior_std: float = 1.0, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    d = phi_dim(K, D)
    if problem is None:
        return torch.randn(M, d, device=device) * prior_std

    X = problem.X.to(device)
    chol_sz = _chol_size(D)
    diag_idx = [sum(range(i + 1)) + i for i in range(D)]
    X_std = X.std(dim=0).mean().item()
    noise_scale = max(X_std * 0.15, 0.1)
    centers = _kmeans_pp_centers(X, K)

    if K > 1:
        cdiff = centers.unsqueeze(0) - centers.unsqueeze(1)
        cdist = (cdiff ** 2).sum(-1).sqrt()
        cdist.fill_diagonal_(float("inf"))
        nn_dist = cdist.min(dim=1).values
        comp_scale = (nn_dist / 2.0).clamp(min=0.1, max=3.0)
    else:
        comp_scale = torch.ones(K, device=device) * max(X_std, 0.2)

    phi_init = torch.zeros(M, d, device=device)
    for m in range(M):
        pi_tilde = torch.randn(K, device=device) * 0.1
        mu_init = centers + torch.randn(K, D, device=device) * noise_scale
        L_vecs = torch.zeros(K, chol_sz, device=device)
        for k in range(K):
            s = float(comp_scale[k].item())
            L_vecs[k, diag_idx] = s * (1.0 + 0.05 * torch.randn(D, device=device))
        phi_init[m] = pack_phi(pi_tilde, mu_init, L_vecs)
    return phi_init
