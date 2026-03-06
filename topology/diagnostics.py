"""
Topological Diagnostics for PHC.

Computes all scalar diagnostics from FiltrationResult + particle weights.
These are the independent fidelity checks that feed back to the navigator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import math
import torch

from topology.filtration import FiltrationResult


@dataclass
class TopoDiagnostics:
    """
    All topological diagnostics at a single time step.

    Fields used by navigator featurizer — must be fixed-size regardless of K.
    """
    # ── Core counts ───────────────────────────────────────────────
    C_tau: int                          # number of clusters at operational tau
    n_born: int                         # total components ever born (incl. dead)

    # ── C(tau) curve sampled at fixed quantiles ───────────────────
    C_at_quantiles: torch.Tensor        # [n_q] integer counts at tau quantile grid

    # ── Per-cluster stats (padded to C_max) ───────────────────────
    C_max: int
    W_c: torch.Tensor                   # [C_max] cluster masses (0 for absent)
    ESS_c: torch.Tensor                 # [C_max] per-cluster ESS (0 for absent)
    persistence_c: torch.Tensor         # [C_max] component lifetimes (0 for absent)
    cluster_mask: torch.Tensor          # [C_max] bool: 1 if cluster exists

    # ── Global scalars ────────────────────────────────────────────
    H_W: float                          # mass entropy -sum W_c log W_c
    min_ESS_c: float                    # minimum per-cluster ESS
    max_persistence: float              # longest-lived component lifetime
    total_persistence: float            # sum of all lifetimes
    barrier_mean: float                 # mean pairwise barrier height
    global_ESS: float                   # overall ESS = (sum w)^2 / sum w^2

    def to_feature_vector(self, tau_quantiles: List[float]) -> torch.Tensor:
        """
        Convert to fixed-size feature vector for navigator input.

        Format: [C_at_quantiles, log(C_tau+1), H_W, log(global_ESS+1),
                 log(min_ESS_c+1), max_persistence, W_c_padded, log(ESS_c_padded+1)]
        """
        device = self.W_c.device

        feats = []

        # C(tau) at quantile grid: [n_q]
        feats.append(self.C_at_quantiles.float())

        # Scalar diagnostics
        feats.append(torch.tensor([
            math.log(self.C_tau + 1),
            self.H_W,
            math.log(self.global_ESS + 1),
            math.log(self.min_ESS_c + 1),
            self.max_persistence,
            self.total_persistence,
            self.barrier_mean,
        ], device=device))

        # Padded per-cluster stats
        feats.append(self.W_c)               # [C_max]
        feats.append(torch.log(self.ESS_c + 1))   # [C_max]
        feats.append(self.persistence_c)     # [C_max]
        feats.append(self.cluster_mask.float())  # [C_max]

        return torch.cat(feats)

    @property
    def feat_dim(self) -> int:
        n_q = self.C_at_quantiles.shape[0]
        return n_q + 7 + 4 * self.C_max




def compute_diagnostics(
    filtration: FiltrationResult,
    weights: torch.Tensor,     # [M] normalized importance weights
    tau_quantiles: List[float],
    C_max: int = 8,
    ESS_min: float = 5.0,
) -> TopoDiagnostics:
    """
    Compute all topological diagnostics from a filtration result.

    Args:
        filtration: output of LevelSetFiltration.run()
        weights:    [M] normalized importance weights
        tau_quantiles: list of quantile fractions in [0,1] for C(tau) sampling
        C_max:      maximum number of clusters to represent
        ESS_min:    minimum ESS threshold for flagging
    Returns:
        TopoDiagnostics
    """
    M = weights.shape[0]
    device = weights.device

    assignments = filtration.assignments  # [M]
    H_vals = filtration.H_vals            # [M]
    n_clusters = filtration.n_clusters

    # ── C(tau) at quantiles ──────────────────────────────────────────────────
    tau_grid = filtration.tau_grid
    C_curve = filtration.C_tau_curve      # [n_grid]

    C_at_quantiles = torch.zeros(len(tau_quantiles), device=device)
    for qi, q in enumerate(tau_quantiles):
        idx = int(q * (len(tau_grid) - 1))
        C_at_quantiles[qi] = C_curve[idx]

    # ── Per-cluster mass and ESS ──────────────────────────────────────────────
    W_c_list = []
    ESS_c_list = []
    persist_c_list = []

    for c in range(n_clusters):
        mask_c = assignments == c
        if not mask_c.any():
            continue
        w_c = weights[mask_c]

        # Cluster mass
        W_c_val = w_c.sum().item()
        W_c_list.append(W_c_val)

        # Cluster ESS
        ess_c = (w_c.sum() ** 2 / (w_c ** 2).sum().clamp(min=1e-30)).item()
        ESS_c_list.append(ess_c)

    # Persistence for live components (those in barcode with assignments matched)
    persist_vals = filtration.persistence
    persist_c_list = persist_vals[:n_clusters].tolist() if len(persist_vals) >= n_clusters else \
        persist_vals.tolist() + [0.0] * (n_clusters - len(persist_vals))

    # Pad to C_max
    def pad_to(lst, size, val=0.0):
        arr = lst[:size] + [val] * max(0, size - len(lst))
        return torch.tensor(arr, device=device, dtype=torch.float32)

    W_c_padded = pad_to(W_c_list, C_max)
    ESS_c_padded = pad_to(ESS_c_list, C_max)
    persist_padded = pad_to(persist_c_list, C_max)
    cluster_mask = torch.zeros(C_max, device=device, dtype=torch.bool)
    cluster_mask[:min(n_clusters, C_max)] = True

    # Normalize W_c for entropy computation
    W_total = W_c_padded.sum().item()
    if W_total > 1e-30 and n_clusters > 1:
        W_norm = W_c_padded[:n_clusters] / W_total
        H_W = float(-( W_norm * torch.log(W_norm.clamp(min=1e-30))).sum().item())
    elif n_clusters == 1:
        H_W = 0.0
    else:
        H_W = 0.0

    # Global ESS
    global_ESS = float((weights.sum() ** 2 / (weights ** 2).sum().clamp(min=1e-30)).item())

    # Min cluster ESS
    min_ESS_c = float(min(ESS_c_list)) if ESS_c_list else 0.0

    # Persistence stats
    persist_arr = filtration.persistence
    max_persistence = float(persist_arr.max().item()) if persist_arr.numel() > 0 else 0.0
    total_persistence = float(persist_arr[persist_arr != float("inf")].sum().item()) \
        if persist_arr.numel() > 0 else 0.0

    # Barrier height (mean of finite off-diagonal)
    bm = filtration.barrier_matrix
    finite_mask = torch.isfinite(bm) & (bm > 0)
    barrier_mean = float(bm[finite_mask].mean().item()) if finite_mask.any() else 0.0

    return TopoDiagnostics(
        C_tau=n_clusters,
        n_born=filtration.n_born,
        C_at_quantiles=C_at_quantiles,
        C_max=C_max,
        W_c=W_c_padded,
        ESS_c=ESS_c_padded,
        persistence_c=persist_padded,
        cluster_mask=cluster_mask,
        H_W=H_W,
        min_ESS_c=min_ESS_c,
        max_persistence=max_persistence,
        total_persistence=total_persistence,
        barrier_mean=barrier_mean,
        global_ESS=global_ESS,
    )
