from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
import torch

from .gmm_problem import GMMProblem, get_streaming_index_batches, sample_gmm_problem, sample_prior_particles
from .toy_stream import feature_map_from_latent


class EpisodeSampler:
    """
    Lightweight adaptor of the GMM episodic sampler for the latent-assimilation toy repo.

    This keeps the current fixed-latent transport code, but upgrades the synthetic protocol:
    each episode is a fresh GMM problem, with feature observations produced from the latent GMM
    samples through the same nonlinear feature map used by the projector.
    """

    def __init__(
        self,
        K_min: int = 4,
        K_max: int = 4,
        D: int = 2,
        N: int = 512,
        overlap_range: tuple = (0.25, 0.45),
        sigma_scale: float = 1.0,
        T_batches: int = 10,
        batch_size: int = 32,
        n_particles: int = 128,
        prior_std: float = 1.0,
        d_in: int = 16,
        feature_noise: float = 0.12,
        latent_noise: float = 0.03,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ):
        self.K_min = K_min
        self.K_max = K_max
        self.D = D
        self.N = N
        self.overlap_range = overlap_range
        self.sigma_scale = sigma_scale
        self.T_batches = T_batches
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.prior_std = prior_std
        self.d_in = d_in
        self.feature_noise = feature_noise
        self.latent_noise = latent_noise
        self.device = device
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def sample_episode(self, fixed_K: Optional[int] = None) -> Dict:
        K = fixed_K if fixed_K is not None else random.randint(self.K_min, self.K_max)
        overlap = random.uniform(*self.overlap_range)
        problem = sample_gmm_problem(
            K=K,
            D=self.D,
            N=self.N,
            overlap=overlap,
            sigma_scale=self.sigma_scale,
            device=self.device,
        )

        z = problem.X.detach().cpu().numpy().astype(np.float32)
        z = z + np.random.normal(scale=self.latent_noise, size=z.shape).astype(np.float32)
        y = problem.labels_true.detach().cpu().numpy().astype(np.int64)
        x = feature_map_from_latent(z, y, self.d_in, self.feature_noise, np.random.default_rng(self.seed))

        idx_batches = get_streaming_index_batches(problem, self.T_batches, self.batch_size, shuffle=True)
        batches = []
        for t, idx in enumerate(idx_batches):
            ii = idx.detach().cpu().numpy()
            batches.append({
                "x": x[ii],
                "z": z[ii],
                "y": y[ii],
                "t": np.array([t], dtype=np.int64),
            })

        phi_init = sample_prior_particles(
            M=self.n_particles,
            K=K,
            D=self.D,
            problem=problem,
            prior_std=self.prior_std,
            device=self.device,
        )
        return {
            "problem": problem,
            "batches": batches,
            "phi_init": phi_init,
            "K": K,
            "overlap": overlap,
            "x": x,
            "z": z,
            "y": y,
        }
