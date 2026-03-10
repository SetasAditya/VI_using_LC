from __future__ import annotations

import math
from typing import Tuple

import torch


def pairwise_sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1).unsqueeze(0)
    return (x2 + y2 - 2.0 * x @ y.T).clamp_min(0.0)


def weighted_mean_cov(z: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = w / (w.sum() + 1e-8)
    mu = (w[:, None] * z).sum(0)
    xc = z - mu
    cov = (w[:, None, None] * (xc[:, :, None] * xc[:, None, :])).sum(0)
    return mu, cov


def gaussian_kde_logprob(points: torch.Tensor, queries: torch.Tensor, logw: torch.Tensor, sigma: float) -> torch.Tensor:
    d2 = pairwise_sqdist(queries, points)
    log_kernel = -0.5 * d2 / (sigma ** 2)
    c = -points.shape[-1] * 0.5 * math.log(2.0 * math.pi * sigma ** 2)
    return torch.logsumexp(logw[None, :] + log_kernel, dim=1) + c


def ess_from_logw(logw: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(logw, dim=0)
    return 1.0 / (w.square().sum() + 1e-8)


def systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    n = weights.numel()
    positions = (torch.arange(n, device=weights.device, dtype=weights.dtype) + torch.rand((), device=weights.device, dtype=weights.dtype)) / n
    cumsum = torch.cumsum(weights, dim=0)
    cumsum[-1] = 1.0
    idx = torch.searchsorted(cumsum, positions, right=False)
    return idx.long()


def sinkhorn_cost(x: torch.Tensor, a: torch.Tensor, y: torch.Tensor, b: torch.Tensor, epsilon: float = 0.25, n_iters: int = 40) -> torch.Tensor:
    a = a / (a.sum() + 1e-8)
    b = b / (b.sum() + 1e-8)
    C = pairwise_sqdist(x, y)
    K = torch.exp(-C / max(epsilon, 1e-6)).clamp_min(1e-12)
    u = torch.full_like(a, 1.0 / max(a.numel(), 1))
    v = torch.full_like(b, 1.0 / max(b.numel(), 1))
    for _ in range(n_iters):
        u = a / (K @ v + 1e-8)
        v = b / (K.t() @ u + 1e-8)
    P = u[:, None] * K * v[None, :]
    return (P * C).sum()


def sinkhorn_divergence(x: torch.Tensor, a: torch.Tensor, y: torch.Tensor, b: torch.Tensor, epsilon: float = 0.25, n_iters: int = 40) -> torch.Tensor:
    c_xy = sinkhorn_cost(x, a, y, b, epsilon=epsilon, n_iters=n_iters)
    c_xx = sinkhorn_cost(x, a, x, a, epsilon=epsilon, n_iters=n_iters)
    c_yy = sinkhorn_cost(y, b, y, b, epsilon=epsilon, n_iters=n_iters)
    return c_xy - 0.5 * c_xx - 0.5 * c_yy
