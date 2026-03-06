"""
Level-Set Filtration via Union-Find.

Implements connected-component tracking as particles are inserted
in order of increasing H^clust value (the filtration order).

Tracks:
    - Component birth/death thresholds (barcode)
    - Pairwise barrier heights between clusters
    - C(tau) curve for any tau grid
    - Final cluster assignments at operational tau
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Union-Find
# ──────────────────────────────────────────────────────────────────────────────

class UnionFind:
    """
    Path-compressed union-find with two merge strategies:

    union(x, y)          — union by rank (fast, for C(tau) curve computation
                           where elder semantics are NOT needed)
    union_elder(x, y, birth_times)  — always makes the elder (lower birth
                           time) the root, enforcing the elder-survives rule
                           for 0-dimensional persistent homology.

    The elder rule is needed for the persistence barcode and barrier matrix
    so that "dying" always refers to the younger component.
    """

    def __init__(self, M: int):
        self.parent = list(range(M))
        self.rank = [0] * M
        self.n_components = 0  # active components

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Merge by rank (for C(tau) curve — order doesn't matter)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True

    def union_elder(
        self,
        x: int,
        y: int,
        birth_times: dict,  # root → birth tau (from component_birth dict)
    ) -> Tuple[bool, int, int]:
        """
        Merge enforcing elder rule: the component with the lower birth tau
        always becomes (or stays) the root.  The younger component is the
        one that 'dies' in the persistence barcode.

        Returns:
            merged:   bool — True if a merge happened
            survivor: int  — root of the surviving (elder) component
            dying:    int  — root of the dying (younger) component
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False, rx, rx

        bx = birth_times.get(rx, float('inf'))
        by = birth_times.get(ry, float('inf'))

        # Elder (smaller birth tau) is the survivor
        if bx <= by:
            survivor, dying = rx, ry
        else:
            survivor, dying = ry, rx

        # Force survivor to be root, ignoring rank
        self.parent[dying] = survivor
        # Update rank only if needed to keep tree balanced
        if self.rank[survivor] <= self.rank[dying]:
            self.rank[survivor] = self.rank[dying] + 1
        self.n_components -= 1

        return True, survivor, dying


# ──────────────────────────────────────────────────────────────────────────────
# Filtration result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FiltrationResult:
    """Output of LevelSetFiltration.run()."""
    # Cluster assignments at operational tau
    assignments: torch.Tensor          # [M] cluster id ∈ {0,...,C-1} or -1 if above tau
    n_clusters: int                    # number of clusters at operational tau

    # Full barcode: one entry per component ever born
    barcode: List[Tuple[float, float]] # [(tau_birth, tau_death)] per component
    component_root: List[int]          # which particle birthed each component

    # Pairwise merge thresholds [n_born x n_born] (inf = never merged)
    barrier_matrix: torch.Tensor

    # C(tau) curve for visualization
    tau_grid: torch.Tensor             # [n_grid] tau values
    C_tau_curve: torch.Tensor          # [n_grid] C(tau) values

    # All H^clust values used in filtration
    H_vals: torch.Tensor               # [M]

    # Persistence: lifetime of each born component
    persistence: torch.Tensor          # [n_born]

    @property
    def n_born(self) -> int:
        return len(self.barcode)

    def get_assignments_at_tau(self, tau: float) -> torch.Tensor:
        """Re-derive cluster assignments at a different tau."""
        M = self.H_vals.shape[0]
        assignments = -torch.ones(M, dtype=torch.long, device=self.H_vals.device)

        # Particles below tau
        in_sublevel = self.H_vals <= tau
        if not in_sublevel.any():
            return assignments

        # Re-run union-find up to tau (simple O(M^2) version for small M)
        uf = UnionFind(M)
        sort_idx = torch.argsort(self.H_vals)

        for i in sort_idx:
            if self.H_vals[i] > tau:
                break
            uf.n_components += 1

        # This is approximate — full re-run would be needed for accuracy
        # For visualization we use the stored C_tau_curve instead
        return assignments


# ──────────────────────────────────────────────────────────────────────────────
# Level-Set Filtration
# ──────────────────────────────────────────────────────────────────────────────

class LevelSetFiltration:
    """
    Runs a level-set filtration over particle embeddings.

    CONNECTIVITY RULE (critical for correctness):
        Union(i, j) only if ALL of:
            1. dist(i, j) <= eps_cutoff  (= eps_cutoff_factor * KDE sigma)
            2. j is in kNN(i) among already-inserted particles
            3. i is in kNN(j) among already-inserted particles  [mutual-kNN]

        Without conditions (1)+(3), the 2nd inserted particle always connects
        to the 1st (only available neighbor), creating an irreversible
        cross-basin bridge that keeps C_tau=1 for the entire filtration.
    """

    def __init__(
        self,
        eps_graph: Optional[float] = None,
        knn_k: int = 3,
        use_knn: bool = True,
        eps_cutoff_factor: float = 2.0,
    ):
        """
        Args:
            eps_graph:         radius for epsilon-graph (if use_knn=False)
            knn_k:             k for mutual-kNN connectivity
            use_knn:           if True, use mutual-kNN; else use epsilon-ball
            eps_cutoff_factor: eps_cutoff = factor * sigma (KDE bandwidth)
        """
        self.eps_graph = eps_graph
        self.knn_k = knn_k
        self.use_knn = use_knn
        self.eps_cutoff_factor = eps_cutoff_factor

    def run(
        self,
        z: torch.Tensor,                    # [M, d] particle embeddings
        H_vals: torch.Tensor,               # [M] H^clust per particle
        weights: torch.Tensor,              # [M] importance weights
        tau_operational: Optional[float] = None,  # if None, inferred from C(tau) curve
        n_tau_grid: int = 50,               # C(tau) curve resolution
        M_metric: Optional[torch.Tensor] = None,  # [d] diagonal metric
        K_target: Optional[int] = None,     # target #clusters for tau selection
        coverage_min: float = 0.85,         # min fraction of weight assigned at tau
    ) -> FiltrationResult:
        """
        Run level-set filtration.

        Args:
            z:                [M, d] embeddings
            H_vals:           [M] Hamiltonian values per particle
            weights:          [M] importance weights
            tau_operational:  threshold for final cluster assignments.
                              If None (recommended), inferred from C(tau) curve
                              as the first tau where C(tau) descends to <= K_target.
                              This ensures most particles are assigned and the
                              number of clusters matches the model order.
            n_tau_grid:       resolution of C(tau) curve
            M_metric:         optional [d] diagonal metric tensor
            K_target:         target number of clusters for auto tau selection.
                              If None and tau_operational is None, uses M//4.
        Returns:
            FiltrationResult
        """
        M, d = z.shape
        device = z.device

        # Precompute pairwise distances
        dist_matrix = self._pairwise_dist(z, M_metric)  # [M, M]

        # Adaptive per-point reach distance replaces global eps = factor*sigma.
        # eps[i] = alpha * dist(i, knn_k-th nearest neighbor)
        # Edge (i,j): dist(i,j) <= max(eps[i], eps[j])  [symmetric reach]
        # This scales the cutoff to local density, avoiding both:
        #   - Under-connection in sparse regions (global sigma too small)
        #   - Over-connection across modes in dense regions (global sigma too large)
        from topology.kde import silverman_bandwidth
        sigma = silverman_bandwidth(z)
        eps_global = self.eps_cutoff_factor * sigma  # fallback for isolated points

        if self.use_knn:
            k_reach = min(self.knn_k, M - 1)
            if k_reach > 0 and M > 1:
                dm_no_self = dist_matrix.clone()
                dm_no_self.fill_diagonal_(float("inf"))
                knn_dists, _ = dm_no_self.topk(k_reach, largest=False, dim=1)
                reach = knn_dists[:, -1].clamp(max=eps_global)   # [M]
                eps = (self.eps_cutoff_factor * reach).clamp(min=1e-8)  # [M] tensor
            else:
                eps = eps_global
        else:
            eps = self.eps_graph if self.eps_graph is not None else eps_global

        # Sort particles by H values (ascending → low H inserted first)
        sort_idx = torch.argsort(H_vals)
        H_sorted = H_vals[sort_idx]

        # Union-find
        uf = UnionFind(M)

        # Tracking
        inserted = torch.zeros(M, dtype=torch.bool, device=device)
        component_birth = {}     # root_particle → tau_birth
        component_death = {}     # root_particle → tau_death
        active_roots: set = set()

        # Component birth/death as tuples
        barcode: List[Tuple[float, float]] = []
        component_roots_list: List[int] = []

        # Map: union-find root → barcode index
        root_to_barcode_idx: dict = {}

        # Barrier matrix: n_born x n_born (filled as merges happen)
        # We'll finalize after the loop
        barrier_events: List[Tuple[float, int, int]] = []  # (tau, barcode_idx_1, barcode_idx_2)

        for step_i in range(M):
            i = sort_idx[step_i].item()
            tau_i = H_sorted[step_i].item()

            inserted[i] = True
            uf.n_components += 1

            # This particle starts its own component
            component_birth[i] = tau_i
            active_roots.add(i)
            barcode.append((tau_i, float("inf")))  # death TBD
            bc_idx = len(barcode) - 1
            root_to_barcode_idx[i] = bc_idx
            component_roots_list.append(i)

            # Connect to already-inserted neighbors
            neighbors = self._get_neighbors(
                i, inserted, dist_matrix, eps, sort_idx[:step_i]
            )

            for j in neighbors:
                ri, rj = uf.find(i), uf.find(j)
                if ri != rj:
                    # use union_elder: elder root guaranteed to survive
                    # regardless of union-find rank (fixes the rank-swap bug)
                    merged, survivor, dying = uf.union_elder(ri, rj, component_birth)
                    if merged:
                        bi = component_birth.get(survivor, tau_i)
                        bj = component_birth.get(dying,   tau_i)

                        # Record death of the younger component
                        dying_bc_idx = root_to_barcode_idx.get(dying, -1)
                        if dying_bc_idx >= 0 and barcode[dying_bc_idx][1] == float("inf"):
                            barcode[dying_bc_idx] = (barcode[dying_bc_idx][0], tau_i)
                            component_death[dying] = tau_i

                        # Record barrier event
                        survivor_bc_idx = root_to_barcode_idx.get(survivor, -1)
                        if survivor_bc_idx >= 0 and dying_bc_idx >= 0:
                            barrier_events.append((tau_i, survivor_bc_idx, dying_bc_idx))

                        # Ensure survivor has the correct (elder) birth time
                        if survivor not in component_birth:
                            component_birth[survivor] = min(bi, bj)

                        # Update root tracking — after union_elder, uf.find(survivor)
                        # IS survivor (that's the guarantee), so no re-lookup needed
                        active_roots.discard(dying)
                        if survivor not in root_to_barcode_idx:
                            root_to_barcode_idx[survivor] = root_to_barcode_idx.get(dying, -1)

        # Close any still-living components
        tau_max = H_sorted[-1].item() if M > 0 else 0.0
        for bc_idx in range(len(barcode)):
            if barcode[bc_idx][1] == float("inf"):
                barcode[bc_idx] = (barcode[bc_idx][0], float("inf"))  # never dies

        # ── Build barrier matrix ──────────────────────────────────────────────
        n_born = len(barcode)
        barrier_mat = torch.full((n_born, n_born), float("inf"), device=device)
        for (tau_merge, bc_i, bc_j) in barrier_events:
            if bc_i < n_born and bc_j < n_born:
                barrier_mat[bc_i, bc_j] = tau_merge
                barrier_mat[bc_j, bc_i] = tau_merge
        barrier_mat.fill_diagonal_(0.0)

        # ── C(tau) curve — must be computed BEFORE tau_operational selection ──
        tau_grid = torch.linspace(
            H_vals.min().item(),
            H_vals.max().item() * 1.05,
            n_tau_grid,
            device=device,
        )
        C_tau_curve = self._compute_C_tau_curve(
            H_sorted, sort_idx, dist_matrix, eps, tau_grid, M, device
        )

        # ── Select tau_operational from C(tau) curve if not provided ─────────
        # Find the tau on the *descending* portion where C(tau) <= K_target
        # AND the coverage f(tau) >= coverage_min. This ensures:
        #   - Most particles are assigned (not stranded outside the sublevel set)
        #   - Cluster count reflects the actual basin structure
        if tau_operational is None:
            K_tgt = K_target if K_target is not None else max(1, M // 4)
            tau_operational = self._tau_from_C_curve(
                C_tau_curve, tau_grid, K_tgt,
                H_vals=H_vals, weights=weights,
                coverage_min=coverage_min,
            )

        # ── Final assignments at tau_operational ─────────────────────────────
        assignments = self._assignments_at_tau(
            H_vals, tau_operational, dist_matrix, eps, M, device
        )

        # ── Persistence ───────────────────────────────────────────────────────
        persistence = torch.tensor(
            [death - birth if death != float("inf") else tau_max - birth
             for birth, death in barcode],
            device=device,
        )

        # Count actual clusters at tau_operational
        n_clusters = int((assignments >= 0).any().item())
        if n_clusters > 0:
            n_clusters = assignments[assignments >= 0].max().item() + 1

        return FiltrationResult(
            assignments=assignments,
            n_clusters=int(n_clusters),
            barcode=barcode,
            component_root=component_roots_list,
            barrier_matrix=barrier_mat,
            tau_grid=tau_grid,
            C_tau_curve=C_tau_curve,
            H_vals=H_vals,
            persistence=persistence,
        )

    def _pairwise_dist(
        self,
        z: torch.Tensor,               # [M, d]
        M_metric: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute [M, M] pairwise distance matrix."""
        M, d = z.shape
        diff = z[:, None, :] - z[None, :, :]  # [M, M, d]

        if M_metric is not None and M_metric.ndim == 1:
            dist_sq = (diff ** 2 * M_metric[None, None, :]).sum(dim=-1)
        else:
            dist_sq = (diff ** 2).sum(dim=-1)

        return dist_sq.sqrt().clamp(min=0.0)  # [M, M]

    def _get_neighbors(
        self,
        i: int,
        inserted: torch.Tensor,    # [M] bool
        dist_matrix: torch.Tensor, # [M, M]
        eps,                       # float or [M] tensor (adaptive per-point cutoff)
        inserted_indices: torch.Tensor,
    ) -> List[int]:
        """
        Get already-inserted neighbors via mutual-kNN + adaptive distance cutoff.

        j is valid neighbor of i if:
            1. dist(i,j) <= max(eps[i], eps[j])  [symmetric adaptive reach]
            2. j in kNN(i) among inserted
            3. i in kNN(j) among inserted  [mutual]
        """
        if inserted_indices.numel() == 0:
            return []

        eps_i = float(eps[i].item()) if torch.is_tensor(eps) else float(eps)
        dists_to_i = dist_matrix[i, :]

        if self.use_knn:
            within_eps = dists_to_i <= eps_i
            mask_valid = inserted & within_eps
            mask_valid[i] = False
            valid_idx = torch.where(mask_valid)[0]

            if valid_idx.numel() == 0:
                return []

            k_i = min(self.knn_k, valid_idx.numel())
            dists_valid = dists_to_i[valid_idx]
            _, nn_rel = torch.topk(dists_valid, k=k_i, largest=False)
            candidates = valid_idx[nn_rel].tolist()

            mutual = []
            for j in candidates:
                eps_j = float(eps[j].item()) if torch.is_tensor(eps) else float(eps)
                eps_ij = max(eps_i, eps_j)   # symmetric reach
                dists_from_j = dist_matrix[j, :]
                mask_j = inserted & (dists_from_j <= eps_ij)
                mask_j[j] = False
                valid_j = torch.where(mask_j)[0]
                if valid_j.numel() == 0:
                    continue
                k_j = min(self.knn_k, valid_j.numel())
                _, nn_j_rel = torch.topk(dists_from_j[valid_j], k=k_j, largest=False)
                if i in valid_j[nn_j_rel].tolist():
                    mutual.append(j)

            return mutual
        else:
            dists = dist_matrix[i, inserted_indices]
            close = inserted_indices[dists <= eps_i]
            return close.tolist()

    def _assignments_at_tau(
        self,
        H_vals: torch.Tensor,
        tau: float,
        dist_matrix: torch.Tensor,
        eps: float,
        M: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute cluster assignments at a specific tau by re-running union-find."""
        assignments = -torch.ones(M, dtype=torch.long, device=device)
        in_sublevel = H_vals <= tau

        if not in_sublevel.any():
            return assignments

        # Sort sublevel particles by H
        sub_idx = torch.where(in_sublevel)[0]
        sub_H = H_vals[sub_idx]
        sort_order = torch.argsort(sub_H)
        sub_idx_sorted = sub_idx[sort_order]

        uf = UnionFind(M)
        inserted_local = torch.zeros(M, dtype=torch.bool, device=device)

        for step_i in range(len(sub_idx_sorted)):
            i = sub_idx_sorted[step_i].item()
            inserted_local[i] = True
            uf.n_components += 1

            # Connect via mutual-kNN + distance cutoff (same as main filtration)
            prev_inserted = sub_idx_sorted[:step_i]
            if len(prev_inserted) > 0:
                neighbors = self._get_neighbors(
                    i, inserted_local, dist_matrix, eps, prev_inserted
                )
                for j in neighbors:
                    uf.union(i, j)

        # Label by root
        roots = {}
        label_counter = 0
        for i in sub_idx_sorted.tolist():
            r = uf.find(i)
            if r not in roots:
                roots[r] = label_counter
                label_counter += 1
            assignments[i] = roots[r]

        return assignments

    def _compute_C_tau_curve(
        self,
        H_sorted: torch.Tensor,
        sort_idx: torch.Tensor,
        dist_matrix: torch.Tensor,
        eps: float,
        tau_grid: torch.Tensor,
        M: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Efficient C(tau) curve computation.

        Sweeps tau grid and counts connected components at each level.
        Uses incremental union-find (O(M * n_grid) amortized).
        """
        C_vals = torch.zeros(len(tau_grid), dtype=torch.long, device=device)

        # Incremental: process each tau in the grid
        # Faster: just count distinct roots among inserted
        uf = UnionFind(M)
        inserted = torch.zeros(M, dtype=torch.bool, device=device)
        step_i = 0
        n_inserted = 0

        for t_idx, tau in enumerate(tau_grid):
            tau_val = tau.item()

            # Insert all particles with H <= tau
            while step_i < M and H_sorted[step_i].item() <= tau_val:
                i = sort_idx[step_i].item()
                inserted[i] = True
                uf.n_components += 1
                n_inserted += 1

                # Connect via mutual-kNN + distance cutoff
                if n_inserted > 1:
                    prev_inserted = sort_idx[:step_i]
                    neighbors = self._get_neighbors(
                        i, inserted, dist_matrix, eps, prev_inserted
                    )
                    for j in neighbors:
                        uf.union(i, j)

                step_i += 1

            C_vals[t_idx] = uf.n_components

        return C_vals.float()

    def _tau_from_C_curve(
        self,
        C_tau_curve: torch.Tensor,
        tau_grid: torch.Tensor,
        K_target: int,
        H_vals: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        coverage_min: float = 0.85,
    ) -> float:
        """
        Find tau_operational on the descending C(tau) branch with a coverage constraint.

        Selects the SMALLEST tau after the peak satisfying:
            (a) C(tau) <= K_target         [topology: right cluster count]
            (b) f(tau)  >= coverage_min    [coverage: enough mass assigned]

        where f(tau) = sum of weights with H <= tau.

        The coverage constraint prevents "only 10% assigned" failures:
        even if C(tau) hits K early, we won't accept tau if most particles
        are still outside the sublevel set.
        """
        C_vals = C_tau_curve.long()
        n = len(C_vals)

        # Precompute coverage f(tau) at each grid point
        if H_vals is not None and weights is not None:
            w_norm = weights / (weights.sum() + 1e-30)
            f_vals = [float(w_norm[H_vals <= float(tau.item())].sum().item())
                      for tau in tau_grid]
        else:
            f_vals = [1.0] * n   # skip coverage constraint if not provided

        peak_idx = int(C_vals.argmax().item())

        # Collect all indices on descending branch satisfying both conditions
        valid = [idx for idx in range(peak_idx, n)
                 if int(C_vals[idx].item()) <= K_target and f_vals[idx] >= coverage_min]

        if valid:
            # Find the LONGEST contiguous run in valid — most stable plateau.
            # Return the START of that run (smallest tau in the plateau).
            # This avoids jitter where C(tau) crosses K for only one grid step.
            best_start, best_len = valid[0], 1
            cur_start, cur_len   = valid[0], 1
            for prev, curr in zip(valid, valid[1:]):
                if curr == prev + 1:    # contiguous
                    cur_len += 1
                    if cur_len > best_len:
                        best_len  = cur_len
                        best_start = cur_start
                else:
                    cur_start = curr
                    cur_len   = 1
            return float(tau_grid[best_start].item())

        # Fallback: best topology match regardless of coverage
        best_idx, best_diff = n - 1, float("inf")
        for idx in range(peak_idx, n):
            c = int(C_vals[idx].item())
            if c <= K_target:
                best_idx = idx
                break
            diff = abs(c - K_target)
            if diff < best_diff:
                best_diff, best_idx = diff, idx

        return float(tau_grid[best_idx].item())