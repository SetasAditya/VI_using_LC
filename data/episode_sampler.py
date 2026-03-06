"""
Episode Sampler.

Generates a stream of synthetic GMM problem instances for episodic training.
Each call to sample_episode() returns a fresh problem with randomized K, D,
overlap, and data, along with pre-partitioned streaming batches.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch

from data.gmm_problem import (
    GMMProblem,
    get_streaming_batches,
    sample_gmm_problem,
    sample_prior_particles,
)


class EpisodeSampler:
    """
    Samples training episodes for the episodic navigator training loop.

    Each episode contains:
        - A randomly drawn GMMProblem (K, D, overlap all randomized)
        - Streaming batches partitioned from problem.X
        - Initial particles sampled from prior
    """

    def __init__(
        self,
        K_min: int = 2,
        K_max: int = 6,
        D: int = 2,
        N: int = 500,
        overlap_range: tuple = (0.2, 0.6),
        sigma_scale: float = 1.0,
        T_batches: int = 10,
        batch_size: int = 50,
        n_particles: int = 64,
        prior_std: float = 1.0,
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
        self.device = device

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def sample_episode(self, fixed_K: Optional[int] = None) -> Dict:
        """
        Sample a single training episode.

        Args:
            fixed_K: if provided, use this K instead of sampling

        Returns:
            dict with:
                'problem':  GMMProblem instance
                'batches':  List[Tensor] of T streaming batches
                'phi_init': [M, phi_dim] initial particles
                'K':        number of components
        """
        # Sample K
        K = fixed_K if fixed_K is not None else random.randint(self.K_min, self.K_max)

        # Sample overlap (controls difficulty)
        overlap = random.uniform(*self.overlap_range)

        # Generate problem
        problem = sample_gmm_problem(
            K=K,
            D=self.D,
            N=self.N,
            overlap=overlap,
            sigma_scale=self.sigma_scale,
            device=self.device,
        )

        # Partition into streaming batches
        batches = get_streaming_batches(
            problem=problem,
            T_batches=self.T_batches,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Initialize particles from data-informed prior
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
        }

    def sample_curriculum_episode(self, episode_idx: int, n_total: int) -> Dict:
        """
        Curriculum version: start with easy problems (small K, low overlap)
        and gradually increase difficulty.

        Args:
            episode_idx: current episode index
            n_total: total number of training episodes
        Returns:
            episode dict
        """
        progress = min(1.0, episode_idx / (n_total * 0.7))  # reach full difficulty at 70%

        # Curriculum: K grows from K_min to K_max
        K_curr = self.K_min + int(progress * (self.K_max - self.K_min))
        K = random.randint(self.K_min, K_curr)

        # Curriculum: overlap decreases (harder) over time
        overlap_max = self.overlap_range[1]
        overlap_min = self.overlap_range[0]
        overlap_curr_min = overlap_max - progress * (overlap_max - overlap_min)
        overlap = random.uniform(overlap_curr_min, overlap_max)

        problem = sample_gmm_problem(
            K=K,
            D=self.D,
            N=self.N,
            overlap=overlap,
            sigma_scale=self.sigma_scale,
            device=self.device,
        )

        batches = get_streaming_batches(
            problem=problem,
            T_batches=self.T_batches,
            batch_size=self.batch_size,
        )

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
            "difficulty": progress,
        }
