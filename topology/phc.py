"""
Progressive Hamiltonian Clustering (PHC) Pipeline.

Orchestrates: Embed → KDE → Filtration → Diagnostics
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from models.gmm_embedder import GMMEmbedder
from topology.kde import cluster_hamiltonian, silverman_bandwidth
from topology.filtration import LevelSetFiltration, FiltrationResult
from topology.diagnostics import TopoDiagnostics, compute_diagnostics


class PHC:
    """
    Progressive Hamiltonian Clustering.

    Runs the full pipeline at each streaming batch:
        F. Embed:     phi → z
        G. Density:   H^clust = -log KDE(z)
        H. Hierarchy: level-set filtration (union-find)
        I. Diagnostics: C_tau, ESS_c, H_W, persistence, barriers
    """

    def __init__(
        self,
        embedder: GMMEmbedder,
        K: int,
        D: int,
        sigma_rule: str = "silverman",
        C_max: int = 8,
        tau_quantiles: Optional[List[float]] = None,
        ESS_min: float = 5.0,
        knn_k: int = 3,
        tau_operational_quantile: float = 0.1,
        n_tau_grid: int = 50,
        eps_cutoff_factor: float = 4.0,
    ):
        self.embedder = embedder
        self.K = K
        self.D = D
        self.sigma_rule = sigma_rule
        self.C_max = C_max
        self.tau_quantiles = tau_quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.ESS_min = ESS_min
        self.tau_operational_quantile = tau_operational_quantile
        self.n_tau_grid = n_tau_grid

        self.filtration = LevelSetFiltration(
            knn_k=knn_k,
            use_knn=True,
            eps_cutoff_factor=eps_cutoff_factor,
        )

    def run(
        self,
        phi: torch.Tensor,                      # [M, phi_dim]
        weights: torch.Tensor,                  # [M] normalized weights
        M_metric: Optional[torch.Tensor] = None, # [d] diagonal metric from navigator
        X_ref: Optional[torch.Tensor] = None,    # [B, D] data batch for responsibility embed
    ) -> Tuple[TopoDiagnostics, FiltrationResult]:
        """
        Run full PHC pipeline.

        Args:
            phi:      [M, phi_dim] current particles
            weights:  [M] normalized importance weights
            M_metric: [d] diagonal mass matrix (for geometry-aware distances)
            X_ref:    [B, D] optional data batch — if provided, uses responsibility
                      embedding which is informative even after parameter convergence
        Returns:
            (TopoDiagnostics, FiltrationResult)
        """
        # F. Embed — use mu-only embedding for PHC:
        #
        #    WHY NOT resp_only: a correctly-fitting K-component particle evaluates
        #    mean responsibility r_k = 1/K for ALL k on balanced data (by symmetry).
        #    Every good particle maps to the CENTER of Δ^{K-1}, not to corners.
        #    PHC sees one cluster = C_tau=1, always, regardless of hypothesis diversity.
        #    Resp_only only separates DEGENERATE particles from good ones.
        #
        #    WHY mu_only: after canonicalization (Hungarian matching), each particle's
        #    component means are in a consistent ordering. Different parameter hypotheses
        #    (e.g., "comp0 near data cluster A" vs "comp0 near data cluster B") give
        #    different mu vectors → separable clusters in K*D dimensional space.
        #    K*D = 8 for K=4, D=2: Silverman σ in 8D with M=64 is ~0.6×std, tight
        #    enough to resolve modes without merging them.
        resp_only = False   # resp_only is wrong; mu_only is set inside embed()
        z = self.embedder.embed(phi, X_ref=X_ref, resp_only=resp_only, mu_only=True)  # [M, K*D]

        # G. Cluster Hamiltonian
        H_vals = cluster_hamiltonian(
            z=z,
            weights=weights,
            sigma=None,  # auto
            M_metric=M_metric,
            sigma_rule=self.sigma_rule,
        )  # [M]

        # Determine operational tau from C(tau) curve (curve-based selection).
        # We pass K_target=K so the filtration finds the tau where C(tau) ≈ K
        # on the descending portion — ensuring most particles are assigned and
        # the cluster count matches the model order.
        # tau_operational=None triggers this auto-selection inside filtration.run().

        # H. Filtration
        result = self.filtration.run(
            z=z,
            H_vals=H_vals,
            weights=weights,
            tau_operational=None,   # auto-selected from C(tau) curve
            n_tau_grid=self.n_tau_grid,
            M_metric=M_metric,
            K_target=self.K,        # target = model order K
        )

        # I. Diagnostics
        diag = compute_diagnostics(
            filtration=result,
            weights=weights,
            tau_quantiles=self.tau_quantiles,
            C_max=self.C_max,
            ESS_min=self.ESS_min,
        )

        return diag, result

    @property
    def feat_dim(self) -> int:
        """Feature dimension output by TopoDiagnostics.to_feature_vector()."""
        n_q = len(self.tau_quantiles)
        return n_q + 7 + 4 * self.C_max 