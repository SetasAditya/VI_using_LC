from __future__ import annotations

import torch
import torch.nn as nn

from .utils import pairwise_sqdist


def soft_cluster_summary(z: torch.Tensor, w: torch.Tensor, num_clusters: int, tau: float = 0.8, iters: int = 4):
    probs = w / (w.sum() + 1e-8)
    idx = torch.argsort(probs, descending=True)[:num_clusters]
    centers = z[idx].clone()

    for _ in range(iters):
        d2 = pairwise_sqdist(z, centers)
        a = torch.softmax(-d2 / tau, dim=1)
        mass = (probs[:, None] * a).sum(0) + 1e-6
        centers = ((probs[:, None] * a)[:, :, None] * z[:, None, :]).sum(0) / mass[:, None]

    d2 = pairwise_sqdist(z, centers)
    a = torch.softmax(-d2 / tau, dim=1)
    mass = (probs[:, None] * a).sum(0) + 1e-6
    centers = ((probs[:, None] * a)[:, :, None] * z[:, None, :]).sum(0) / mass[:, None]
    cov_trace = (((z[:, None, :] - centers[None, :, :]) ** 2).sum(-1) * probs[:, None] * a).sum(0) / mass
    return mass, centers, cov_trace, a


def gaussian_level_function(query: torch.Tensor, support: torch.Tensor, weights: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Astolfi-style level function H = sum_i w_i exp(-||z-z_i||^2 / (2 h^2))."""
    weights = weights / (weights.sum() + 1e-8)
    d2 = pairwise_sqdist(query, support)
    ker = torch.exp(-0.5 * d2 / max(float(bandwidth) ** 2, 1e-8))
    return ker @ weights


def gaussian_level_gradient(query: torch.Tensor, support: torch.Tensor, weights: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Gradient of the Gaussian level function with respect to query."""
    h2 = max(float(bandwidth) ** 2, 1e-8)
    weights = weights / (weights.sum() + 1e-8)
    diff = support[None, :, :] - query[:, None, :]
    d2 = (diff ** 2).sum(dim=-1)
    ker = torch.exp(-0.5 * d2 / h2) * weights[None, :]
    grad = (ker[:, :, None] * diff).sum(dim=1) / h2
    return grad


def _merge_close_centers(centers: torch.Tensor, scores: torch.Tensor, merge_tol: float, target_k: int):
    if centers.shape[0] == 0:
        return centers, scores
    order = torch.argsort(scores, descending=True)
    kept = []
    kept_scores = []
    for idx in order.tolist():
        c = centers[idx]
        if not kept:
            kept.append(c)
            kept_scores.append(scores[idx])
            continue
        d = torch.stack([torch.norm(c - kc) for kc in kept])
        if torch.min(d) > merge_tol:
            kept.append(c)
            kept_scores.append(scores[idx])
        if len(kept) >= max(int(target_k), 1):
            break
    return torch.stack(kept, dim=0), torch.stack(kept_scores, dim=0)


def hamiltonian_mode_summary(
    z: torch.Tensor,
    w: torch.Tensor,
    num_modes: int,
    bandwidth: float = 0.9,
    iters: int = 6,
    merge_tol: float = 0.55,
    assign_tau: float = 0.45,
):
    """Extract KDE modes of the latent level function via weighted mean-shift.

    Returns a fixed-size summary aligned with the Astolfi notion: density maxima induce
    superlevel-set connected components; here we approximate their centers by KDE modes.
    """
    probs = w / (w.sum() + 1e-8)
    M = z.shape[0]
    K = max(int(num_modes), 1)
    init_k = min(max(2 * K, K), M)
    idx = torch.argsort(probs, descending=True)[:init_k]
    centers = z[idx].clone()
    h2 = max(float(bandwidth) ** 2, 1e-8)

    for _ in range(max(int(iters), 1)):
        d2 = pairwise_sqdist(centers, z)
        ker = torch.exp(-0.5 * d2 / h2) * probs[None, :]
        mass = ker.sum(dim=1, keepdim=True) + 1e-8
        centers = (ker @ z) / mass

    scores = gaussian_level_function(centers, z, probs, bandwidth)
    centers, scores = _merge_close_centers(centers, scores, float(merge_tol), target_k=K)

    if centers.shape[0] < K:
        extra_idx = torch.argsort(probs, descending=True)
        extra = []
        for i in extra_idx.tolist():
            cand = z[i]
            if centers.shape[0] == 0:
                extra.append(cand)
            else:
                d = torch.norm(centers - cand[None, :], dim=1)
                if torch.min(d) > 0.5 * float(merge_tol):
                    extra.append(cand)
            if len(extra) + centers.shape[0] >= K:
                break
        if extra:
            centers = torch.cat([centers, torch.stack(extra, dim=0)], dim=0)
            scores = torch.cat([scores, gaussian_level_function(torch.stack(extra, dim=0), z, probs, bandwidth)], dim=0)
    centers = centers[:K]
    scores = scores[:K]
    if centers.shape[0] < K:
        if centers.shape[0] == 0:
            pad = z[torch.argsort(probs, descending=True)[:K]].clone()
            centers = pad
            scores = gaussian_level_function(centers, z, probs, bandwidth)
        else:
            pad_idx = torch.argsort(probs, descending=True)
            pads = []
            for i in pad_idx.tolist():
                pads.append(z[i])
                if len(pads) >= K - centers.shape[0]:
                    break
            centers = torch.cat([centers, torch.stack(pads, dim=0)], dim=0)
            scores = torch.cat([scores, gaussian_level_function(torch.stack(pads, dim=0), z, probs, bandwidth)], dim=0)
    centers = centers[:K]
    scores = scores[:K]

    d2 = pairwise_sqdist(z, centers)
    logits = -d2 / max(float(assign_tau) ** 2, 1e-6) + torch.log(scores.clamp_min(1e-8))[None, :]
    assign = torch.softmax(logits, dim=1)
    mass = (probs[:, None] * assign).sum(dim=0) + 1e-8
    centers = ((probs[:, None] * assign)[:, :, None] * z[:, None, :]).sum(dim=0) / mass[:, None]
    cov_trace = (((z[:, None, :] - centers[None, :, :]) ** 2).sum(-1) * probs[:, None] * assign).sum(dim=0) / mass
    peak = gaussian_level_function(centers, z, probs, bandwidth)
    entropy = -(mass * (mass + 1e-8).log()).sum()
    eff_modes = torch.exp(entropy)
    sep = torch.cdist(centers, centers)
    sep = sep + torch.eye(centers.shape[0], device=z.device) * 1e6
    min_sep = sep.min(dim=1).values
    return {
        "mass": mass,
        "centers": centers,
        "cov_trace": cov_trace,
        "assignments": assign,
        "peak": peak,
        "effective_modes": eff_modes,
        "min_separation": min_sep,
    }


class ClusterTokenEncoder(nn.Module):
    def __init__(self, token_dim: int = 6, model_dim: int = 64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(token_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim), nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.phi(tokens)
        return self.rho(h.mean(dim=0))
