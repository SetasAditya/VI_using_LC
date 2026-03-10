import math
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class GMM2D:
    weights: np.ndarray
    means: np.ndarray
    stds: np.ndarray

    @property
    def K(self):
        return len(self.weights)

    def sample(self, n: int) -> np.ndarray:
        comp = np.random.choice(self.K, size=n, p=self.weights)
        x = np.zeros((n, 2), dtype=np.float32)
        for k in range(self.K):
            idx = np.where(comp == k)[0]
            if len(idx):
                x[idx] = self.means[k] + np.random.randn(len(idx), 2).astype(np.float32) * self.stds[k]
        return x

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        vals = []
        for k in range(self.K):
            diff = (x - self.means[k]) / self.stds[k]
            lp = -0.5 * np.sum(diff**2, axis=1) - np.sum(np.log(self.stds[k])) - math.log(2 * math.pi)
            vals.append(np.log(self.weights[k] + 1e-12) + lp)
        vals = np.stack(vals, axis=1)
        m = vals.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(vals - m).sum(axis=1, keepdims=True))).squeeze(1)


def make_random_gmm(K: int) -> GMM2D:
    angles = np.linspace(0, 2 * math.pi, K, endpoint=False) + np.random.uniform(-0.35, 0.35, size=K)
    radii = np.random.uniform(2.0, 3.4, size=K)
    means = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1).astype(np.float32)
    means += np.random.randn(K, 2).astype(np.float32) * 0.10
    stds = np.random.uniform(0.20, 0.42, size=(K, 2)).astype(np.float32)
    weights = np.random.dirichlet(np.full(K, 0.9)).astype(np.float32)
    weights = weights / weights.sum()
    return GMM2D(weights=weights, means=means, stds=stds)


def make_task_sequence(
    gmm: GMM2D,
    T_steps: int,
    batch_size_obs: int,
    M_particles: int,
) -> Dict:
    initial_source = (np.random.randn(M_particles, 2).astype(np.float32) * 3.0)

    steps = []
    for k in range(T_steps):
        obs_batch = gmm.sample(batch_size_obs)
        steps.append({
            'obs_batch': obs_batch.astype(np.float32),
            'step_frac': np.array([[(k + 1) / T_steps]], dtype=np.float32),
            'true_means': gmm.means.astype(np.float32),
            'true_weights': gmm.weights.astype(np.float32),
            'true_stds': gmm.stds.astype(np.float32),
        })

    return {
        'initial_source': initial_source,
        'steps': steps,
        'gmm': gmm,
    }


def make_family_sequences(
    num_sequences: int,
    K: int,
    T_steps: int,
    batch_size_obs: int,
    M_particles: int,
):
    family = []
    for _ in range(num_sequences):
        gmm = make_random_gmm(K)
        family.append(make_task_sequence(gmm, T_steps, batch_size_obs, M_particles))
    return family


def _sample_gmm_torch(
    weights: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    B, K = weights.shape
    comp = torch.multinomial(weights, n_samples, replacement=True)
    gather_idx = comp.unsqueeze(-1).expand(-1, -1, 2)
    mu = torch.gather(means, 1, gather_idx)
    sd = torch.gather(stds, 1, gather_idx)
    eps = torch.randn(B, n_samples, 2, device=weights.device, dtype=weights.dtype)
    return mu + sd * eps


@torch.no_grad()
def build_online_endpoint_torch(
    src: torch.Tensor,
    obs_batch: torch.Tensor,
    true_means: torch.Tensor,
    true_weights: torch.Tensor,
    true_stds: torch.Tensor,
    teacher_mult: int = 3,
    teacher_jitter: float = 0.02,
) -> torch.Tensor:
    """
    Build an endpoint cloud independently of the current source geometry.
    The source only determines the number of particles to draw.

    src:          [B, N, 2]
    obs_batch:    [B, O, 2]
    true_means:   [B, K, 2]
    true_weights: [B, K]
    true_stds:    [B, K, 2]
    returns:      [B, N, 2]
    """
    B, N, D = src.shape
    n_teacher = max(teacher_mult * N, obs_batch.shape[1])

    teacher_samples = _sample_gmm_torch(true_weights, true_means, true_stds, n_teacher)
    support = torch.cat([obs_batch, teacher_samples], dim=1)

    S = support.shape[1]
    idx = torch.randint(low=0, high=S, size=(B, N), device=support.device)
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, D)
    target = torch.gather(support, 1, gather_idx)

    if teacher_jitter > 0:
        target = target + teacher_jitter * torch.randn_like(target)
    return target