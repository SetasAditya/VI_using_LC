"""
GMM Problem Generator.

Generates synthetic GMM problem instances for episodic training.
Each episode: sample true (pi, mu, Sigma), generate X, define streaming batches.

State packing convention (critical - must match across all modules):
    phi = [pi_tilde (K,) | mu_1 ... mu_K (K*D,) | L_1 ... L_K (K * D*(D+1)/2,)]
    where:
        pi = softmax(pi_tilde)
        Sigma_k = L_k @ L_k.T  (L_k lower-triangular, entries packed row-major)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.distributions as dist


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _chol_size(D: int) -> int:
    """Number of lower-triangular entries in a D×D matrix."""
    return D * (D + 1) // 2


def phi_dim(K: int, D: int) -> int:
    """Total dimension of particle state phi for given K and D."""
    return K + K * D + K * _chol_size(D)


def pack_phi(
    pi_tilde: torch.Tensor,   # [K]
    mu: torch.Tensor,          # [K, D]
    L_vecs: torch.Tensor,      # [K, D*(D+1)/2]
) -> torch.Tensor:
    """Pack (pi_tilde, mu, L_vecs) into flat phi vector."""
    return torch.cat([pi_tilde, mu.flatten(), L_vecs.flatten()])


def unpack_phi(
    phi: torch.Tensor,  # [phi_dim] or [M, phi_dim]
    K: int,
    D: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpack phi into (pi_tilde, mu, L_vecs).

    Returns:
        pi_tilde: [..., K]
        mu:       [..., K, D]
        L_vecs:   [..., K, D*(D+1)/2]
    """
    chol_sz = _chol_size(D)
    batch = phi.shape[:-1]

    pi_tilde = phi[..., :K]
    mu = phi[..., K:K + K * D].reshape(*batch, K, D)
    L_vecs = phi[..., K + K * D:].reshape(*batch, K, chol_sz)
    return pi_tilde, mu, L_vecs


def L_vec_to_matrix(L_vec: torch.Tensor, D: int) -> torch.Tensor:
    """
    Convert lower-triangular vector (row-major) to D×D matrix.

    Args:
        L_vec: [..., D*(D+1)/2]
    Returns:
        L:     [..., D, D]
    """
    *batch, _ = L_vec.shape
    L = torch.zeros(*batch, D, D, dtype=L_vec.dtype, device=L_vec.device)
    idx = torch.tril_indices(D, D, device=L_vec.device)
    L[..., idx[0], idx[1]] = L_vec
    # Ensure positive diagonal for valid Cholesky
    diag_idx = torch.arange(D, device=L_vec.device)
    L[..., diag_idx, diag_idx] = L[..., diag_idx, diag_idx].abs().clamp(min=1e-3)
    return L


def L_matrix_to_vec(L: torch.Tensor, D: int) -> torch.Tensor:
    """
    Convert D×D lower-triangular matrix to vector (row-major).

    Args:
        L: [..., D, D]
    Returns:
        L_vec: [..., D*(D+1)/2]
    """
    idx = torch.tril_indices(D, D, device=L.device)
    return L[..., idx[0], idx[1]]


# ──────────────────────────────────────────────────────────────────────────────
# GMMProblem dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GMMProblem:
    """A single synthetic GMM problem instance."""
    K: int
    D: int
    pi_true: torch.Tensor        # [K] true mixing weights
    mu_true: torch.Tensor        # [K, D] true means
    Sigma_true: torch.Tensor     # [K, D, D] true covariances
    L_true: torch.Tensor         # [K, D, D] Cholesky factors
    X: torch.Tensor              # [N, D] full dataset
    labels_true: torch.Tensor    # [N] true component assignments
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
        return self.X.shape[0]

    @property
    def phi_dim(self) -> int:
        return phi_dim(self.K, self.D)

    def true_phi(self) -> torch.Tensor:
        """Return ground-truth packed as phi vector (for evaluation)."""
        pi_tilde = torch.log(self.pi_true + 1e-8)  # inverse softmax (up to const)
        L_vecs = L_matrix_to_vec(self.L_true, self.D)  # [K, chol_sz]
        return pack_phi(pi_tilde, self.mu_true, L_vecs)


# ──────────────────────────────────────────────────────────────────────────────
# Problem sampler
# ──────────────────────────────────────────────────────────────────────────────

def sample_gmm_problem(
    K: int,
    D: int,
    N: int,
    overlap: float = 0.3,
    sigma_scale: float = 1.0,
    device: torch.device = torch.device("cpu"),
    rng: Optional[torch.Generator] = None,
) -> GMMProblem:
    """
    Sample a random GMM problem instance.

    Args:
        K: number of components
        D: data dimension
        N: number of data points
        overlap: controls separation (higher = more overlap)
            separation = sigma_scale / overlap
        sigma_scale: scale of true covariances
        device: torch device
        rng: optional torch.Generator for reproducibility
    Returns:
        GMMProblem instance
    """
    separation = sigma_scale / max(overlap, 0.05)

    # Sample mixing weights
    alpha = torch.ones(K, device=device) * 2.0
    pi_true = dist.Dirichlet(alpha).sample() if rng is None else \
        dist.Dirichlet(alpha).sample()
    pi_true = pi_true / pi_true.sum()  # normalize

    # Sample means on a grid-like layout scaled by separation
    mu_true = _sample_separated_means(K, D, separation, device, rng)

    # Sample covariances (random rotation + scaled diagonal)
    L_true = _sample_cholesky_factors(K, D, sigma_scale, device, rng)
    Sigma_true = torch.bmm(L_true, L_true.transpose(-1, -2))  # [K, D, D]

    # Generate data
    X, labels_true = _sample_gmm_data(N, pi_true, mu_true, L_true, device, rng)

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


def _sample_separated_means(
    K: int,
    D: int,
    separation: float,
    device: torch.device,
    rng: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample K means that are at least `separation` apart (best-effort)."""
    means = []
    max_attempts = 1000
    for k in range(K):
        for _ in range(max_attempts):
            mu = torch.randn(D, device=device) * separation
            if not means or all(
                torch.norm(mu - m) >= separation * 0.5 for m in means
            ):
                means.append(mu)
                break
        else:
            # Fall back: just use random placement
            means.append(torch.randn(D, device=device) * separation)
    return torch.stack(means)  # [K, D]


def _sample_cholesky_factors(
    K: int,
    D: int,
    sigma_scale: float,
    device: torch.device,
    rng: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample K random Cholesky factors for D×D positive-definite matrices."""
    Ls = []
    for _ in range(K):
        # Random lower triangular with positive diagonal
        L = torch.zeros(D, D, device=device)
        # Off-diagonal: small random
        for i in range(D):
            for j in range(i):
                L[i, j] = torch.randn(1, device=device).item() * sigma_scale * 0.3
            # Diagonal: log-normal to ensure positivity
            L[i, i] = torch.exp(
                torch.randn(1, device=device) * 0.3
            ).item() * sigma_scale
        Ls.append(L)
    return torch.stack(Ls)  # [K, D, D]


def _sample_gmm_data(
    N: int,
    pi: torch.Tensor,       # [K]
    mu: torch.Tensor,       # [K, D]
    L: torch.Tensor,        # [K, D, D]
    device: torch.device,
    rng: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample N points from a GMM defined by (pi, mu, L)."""
    K, D = mu.shape

    # Sample component assignments
    labels = dist.Categorical(pi).sample((N,))  # [N]

    X = torch.zeros(N, D, device=device)
    for k in range(K):
        mask = labels == k
        n_k = mask.sum().item()
        if n_k > 0:
            eps = torch.randn(n_k, D, device=device)
            X[mask] = mu[k] + (L[k] @ eps.T).T

    return X, labels


# ──────────────────────────────────────────────────────────────────────────────
# Streaming batch construction
# ──────────────────────────────────────────────────────────────────────────────

def get_streaming_batches(
    problem: GMMProblem,
    T_batches: int,
    batch_size: int,
    shuffle: bool = True,
) -> List[torch.Tensor]:
    """
    Partition problem.X into T sequential batches.

    Args:
        problem: GMMProblem instance
        T_batches: number of batches
        batch_size: points per batch
        shuffle: whether to shuffle before partitioning
    Returns:
        list of T tensors, each [batch_size, D]
    """
    N = problem.N
    X = problem.X

    if shuffle:
        perm = torch.randperm(N, device=problem.device)
        X = X[perm]

    batches = []
    for k in range(T_batches):
        start = (k * batch_size) % N
        end = min(start + batch_size, N)
        batch = X[start:end]
        # If we hit end of dataset, wrap around
        if batch.shape[0] < batch_size:
            extra = X[:batch_size - batch.shape[0]]
            batch = torch.cat([batch, extra], dim=0)
        batches.append(batch)

    return batches


# ──────────────────────────────────────────────────────────────────────────────
# Prior over phi
# ──────────────────────────────────────────────────────────────────────────────

def _kmeans_pp_centers(X: torch.Tensor, K: int) -> torch.Tensor:
    """
    K-means++ seeding: greedily pick K centers that are spread across X.

    Returns:
        centers: [K, D]
    """
    N, D = X.shape
    device = X.device

    # Pick first center uniformly at random
    idx0 = torch.randint(N, (1,), device=device).item()
    centers = [X[idx0]]

    for _ in range(1, K):
        # D^2 distance from each point to its nearest center
        c_stack = torch.stack(centers, dim=0)          # [c, D]
        diffs = X.unsqueeze(1) - c_stack.unsqueeze(0)  # [N, c, D]
        dists = (diffs ** 2).sum(-1).min(dim=1).values  # [N]
        dists = dists.clamp(min=0)

        # Sample proportional to D^2
        probs = dists / dists.sum().clamp(min=1e-12)
        idx = torch.multinomial(probs, 1).item()
        centers.append(X[idx])

    return torch.stack(centers, dim=0)   # [K, D]


def sample_prior_particles(
    M: int,
    K: int,
    D: int,
    problem: Optional[GMMProblem] = None,
    prior_std: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Sample M particles from prior over phi.

    Uses K-means++ seeding when problem is provided:
    - Run K-means++ on the full dataset to find K well-separated centers.
    - Each of the M particles is initialized with component means near
      these K centers (+ small noise), so every particle starts with
      all K modes covered in its initial guess.
    - This eliminates the mode-collapse-at-init problem where random
      sampling may pick K data points all near the same 1-2 clusters.

    Returns:
        phi: [M, phi_dim(K, D)]
    """
    d = phi_dim(K, D)

    if problem is None:
        return torch.randn(M, d, device=device) * prior_std

    X = problem.X.to(device)    # [N, D]
    chol_sz = _chol_size(D)
    diag_idx = [sum(range(i + 1)) + i for i in range(D)]

    # ── K-means++ seeding: one set of K centers for all M particles ──────────
    # Each run of K-means++ gives a different spread; we use a single run
    # per call and add per-particle noise to diversify.
    # Estimate component scale from data spread
    X_std = X.std(dim=0).mean().item()          # rough scale of data
    noise_scale = max(X_std * 0.15, 0.1)        # 15% of data spread as init noise

    centers = _kmeans_pp_centers(X, K)          # [K, D]  deterministic-ish seed

    # ── Estimate component scales from nearest-neighbor distances ────────────
    # Used to initialize Cholesky diagonals realistically
    if K > 1:
        cdiff = centers.unsqueeze(0) - centers.unsqueeze(1)   # [K, K, D]
        cdist = (cdiff ** 2).sum(-1).sqrt()                    # [K, K]
        # Each component's "radius" = half its distance to nearest neighbor
        cdist.fill_diagonal_(float('inf'))
        nn_dist = cdist.min(dim=1).values                      # [K]
        comp_scale = (nn_dist / 2.0).clamp(min=0.1, max=3.0)  # [K]
    else:
        comp_scale = torch.ones(K, device=device) * X_std

    phi_init = torch.zeros(M, d, device=device)

    for m in range(M):
        # pi_tilde: small random perturbation around uniform
        pi_tilde = torch.randn(K, device=device) * 0.1

        # mu: centers + independent noise (ensures M particles span different
        # perturbations of the same good initialisation)
        mu_init = centers + torch.randn(K, D, device=device) * noise_scale

        # L: diagonal entries initialized to comp_scale (component-specific)
        L_vecs = torch.zeros(K, chol_sz, device=device)
        for k in range(K):
            s = comp_scale[k].item()
            L_vecs[k, diag_idx] = s * (1.0 + torch.randn(D, device=device) * 0.05)

        phi_init[m] = pack_phi(pi_tilde, mu_init, L_vecs)

    return phi_init
