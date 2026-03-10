from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .energy import KineticEnergy, LatentEnergy, LevelSetMixture, MemoryPolicy, QueryPolicy
from .topology import ClusterTokenEncoder, gaussian_level_function, hamiltonian_mode_summary
from .utils import ess_from_logw, pairwise_sqdist, systematic_resample


class ActiveLatentAssimilator(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden_dim: int = 64, num_particles: int = 128,
                 num_clusters: int = 4, obs_sigma: float = 0.45, teacher_sigma: float = 0.35,
                 cluster_tau: float = 0.7, rejuvenation: float = 0.06, num_classes: int = 4,
                 teacher_move_steps: int = 6, teacher_move_step_size: float = 0.055,
                 student_move_steps: int = 2, student_move_step_size: float = 0.035,
                 ess_resample_frac: float = 0.35, topology_bandwidth: float = 0.90,
                 mode_merge_tol: float = 0.70, energy_residual_weight: float = 0.08,
                 obs_support_weight: float = 0.35, num_level_components: int | None = None,
                 support_level_blend: float = 0.35):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_particles = num_particles
        self.num_clusters = num_clusters
        self.obs_sigma = obs_sigma
        self.teacher_sigma = teacher_sigma
        self.cluster_tau = cluster_tau
        self.rejuvenation = rejuvenation
        self.num_classes = num_classes
        self.teacher_move_steps = teacher_move_steps
        self.teacher_move_step_size = teacher_move_step_size
        self.student_move_steps = student_move_steps
        self.student_move_step_size = student_move_step_size
        self.ess_resample_frac = ess_resample_frac
        self.topology_bandwidth = topology_bandwidth
        self.mode_merge_tol = mode_merge_tol
        self.energy_residual_weight = energy_residual_weight
        self.obs_support_weight = obs_support_weight
        self.support_level_blend = support_level_blend
        self.num_level_components = int(num_level_components or max(2 * num_clusters, num_clusters + 2))

        self.state_token_encoder = ClusterTokenEncoder(token_dim=latent_dim + 5, model_dim=hidden_dim)
        self.coverage_proj = nn.Sequential(
            nn.Linear(2 * num_classes + 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        candidate_dim = 4 * latent_dim + 5 + num_clusters + num_classes + 2
        self.candidate_proj = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.level_head = LevelSetMixture(latent_dim=latent_dim, num_components=self.num_level_components, ctx_dim=hidden_dim, hidden_dim=hidden_dim)
        self.energy = LatentEnergy(ctx_dim=hidden_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.kinetic = KineticEnergy(latent_dim=latent_dim)
        self.policy = MemoryPolicy(in_dim=hidden_dim, hidden_dim=hidden_dim)
        self.query_policy = QueryPolicy(state_dim=hidden_dim, cand_dim=hidden_dim, hidden_dim=hidden_dim)

    def init_state(self, device: torch.device):
        z = 1.4 * torch.randn(self.num_particles, self.latent_dim, device=device)
        p = torch.zeros_like(z)
        logw = torch.full((self.num_particles,), -math.log(self.num_particles), device=device)
        h = torch.zeros(self.hidden_dim, device=device)
        obs_hist = torch.zeros(self.num_classes, device=device)
        return z, p, logw, h, obs_hist

    def learned_level(self, z: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        return self.level_head(z, ctx)

    def learned_log_density(self, z: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        return self.level_head.log_density(z, ctx)

    def learned_component_params(self, ctx: torch.Tensor | None = None):
        return self.level_head.component_params(ctx)

    def _hamiltonian_summary(self, z: torch.Tensor, logw: torch.Tensor) -> Dict[str, torch.Tensor]:
        w = torch.softmax(logw, dim=0)
        return hamiltonian_mode_summary(
            z, w, self.num_clusters,
            bandwidth=self.topology_bandwidth,
            merge_tol=self.mode_merge_tol,
            assign_tau=self.cluster_tau,
        )

    def state_context(self, z: torch.Tensor, logw: torch.Tensor,
                      observed_hist: torch.Tensor | None = None,
                      target_freq: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        summary = self._hamiltonian_summary(z, logw)
        ess = ess_from_logw(logw) / max(float(self.num_particles), 1.0)
        eff_token = torch.full((self.num_clusters, 1), summary["effective_modes"].detach(), device=z.device)
        tokens = torch.cat([
            summary["mass"][:, None],
            summary["centers"],
            summary["cov_trace"][:, None],
            summary["peak"][:, None],
            summary["min_separation"][:, None],
            eff_token,
        ], dim=1)
        topo = self.state_token_encoder(tokens)
        stats = {
            "mass": summary["mass"],
            "centers": summary["centers"],
            "cov_trace": summary["cov_trace"],
            "assignments": summary["assignments"],
            "peak": summary["peak"],
            "effective_modes": summary["effective_modes"].detach(),
            "min_separation": summary["min_separation"].detach(),
            "ess": ess.detach(),
        }
        if observed_hist is None or target_freq is None:
            return topo, stats
        hist_norm = observed_hist / observed_hist.sum().clamp_min(1.0)
        deficit = (target_freq - hist_norm).clamp_min(0.0)
        cover_raw = torch.cat([
            hist_norm,
            deficit,
            torch.tensor([ess.detach(), summary["effective_modes"].detach() / max(float(self.num_clusters), 1.0)], device=z.device),
        ], dim=0)
        cover_ctx = self.coverage_proj(cover_raw)
        stats["hist_norm"] = hist_norm.detach()
        stats["deficit"] = deficit.detach()
        return topo + cover_ctx, stats

    def _score_field(self, z: torch.Tensor, batch_z: torch.Tensor, sigma: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d2 = pairwise_sqdist(z, batch_z)
        attn = torch.softmax(-0.5 * d2 / (sigma ** 2), dim=1)
        obs_target = attn @ batch_z
        drift = (obs_target - z) / max(sigma ** 2, 1e-6)
        ll = torch.logsumexp(-0.5 * d2 / (sigma ** 2), dim=1) - math.log(batch_z.shape[0])
        return drift, obs_target, ll

    def _latent_level_energy(self, query: torch.Tensor, support: torch.Tensor, support_w: torch.Tensor, ctx: torch.Tensor):
        query_req = query.detach().requires_grad_(True)
        learned_level = self.learned_level(query_req, ctx.detach())
        support_level = gaussian_level_function(query_req, support, support_w, self.topology_bandwidth)
        level = (1.0 - self.support_level_blend) * learned_level + self.support_level_blend * support_level
        residual = self.energy(query_req, ctx.detach()) if self.energy_residual_weight > 0.0 else torch.zeros(query_req.shape[0], device=query.device)
        total = -torch.log(level.clamp_min(1e-8)) + self.energy_residual_weight * residual
        grad_total = torch.autograd.grad(total.sum(), query_req, create_graph=False)[0]
        return total.detach(), grad_total.detach(), level.detach(), learned_level.detach(), support_level.detach()

    def candidate_features(self, candidate_z: torch.Tensor, z: torch.Tensor, logw: torch.Tensor,
                           centers: torch.Tensor, target_centroids: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        C, B, D = candidate_z.shape
        mean = candidate_z.mean(dim=1)
        std = candidate_z.std(dim=1)
        q25 = candidate_z.quantile(0.25, dim=1)
        q75 = candidate_z.quantile(0.75, dim=1)
        radius = ((candidate_z - mean[:, None, :]) ** 2).sum(dim=-1).mean(dim=1, keepdim=True).sqrt()
        with torch.no_grad():
            d2 = pairwise_sqdist(candidate_z.reshape(C * B, D), z).reshape(C, B, -1)
            w = torch.softmax(logw, dim=0)
            cand_logpost = torch.logsumexp(-0.5 * d2 / (self.obs_sigma ** 2) + torch.log(w + 1e-8)[None, None, :], dim=-1)
            ll_mean = cand_logpost.mean(dim=1, keepdim=True)
            ll_std = cand_logpost.std(dim=1, keepdim=True)
            d2c = pairwise_sqdist(candidate_z.reshape(C * B, D), centers).reshape(C, B, -1)
            center_assign = torch.softmax(-d2c / max(self.cluster_tau ** 2, 1e-6), dim=-1)
            center_mass = center_assign.mean(dim=1)
            center_dist = torch.cdist(mean, centers).min(dim=1).values[:, None]
            if target_centroids is None:
                target_mass = torch.full((C, self.num_classes), 1.0 / max(self.num_classes, 1), device=z.device)
                target_min = torch.zeros((C, 1), device=z.device)
                target_margin = torch.zeros((C, 1), device=z.device)
            else:
                d2t = pairwise_sqdist(candidate_z.reshape(C * B, D), target_centroids).reshape(C, B, -1)
                target_assign = torch.softmax(-d2t / max(self.cluster_tau ** 2, 1e-6), dim=-1)
                target_mass = target_assign.mean(dim=1)
                td = torch.cdist(mean, target_centroids)
                ttop2 = td.topk(k=min(2, td.shape[1]), largest=False).values
                target_min = ttop2[:, :1]
                target_margin = (ttop2[:, 1:2] - ttop2[:, :1]) if ttop2.shape[1] > 1 else torch.zeros_like(target_min)
            local_summary = []
            for j in range(C):
                s = hamiltonian_mode_summary(candidate_z[j], torch.full((B,), 1.0 / B, device=z.device), self.num_clusters,
                                             bandwidth=self.topology_bandwidth, merge_tol=self.mode_merge_tol, assign_tau=self.cluster_tau)
                local_summary.append(torch.tensor([float(s["effective_modes"].cpu())], device=z.device))
            eff_modes = torch.stack(local_summary, dim=0)
        raw = torch.cat([mean, std, q25, q75, radius, ll_mean, ll_std, center_dist, eff_modes, center_mass, target_mass, target_min, target_margin], dim=1)
        feat = self.candidate_proj(raw)
        return feat, {
            "mean": mean,
            "std": std,
            "q25": q25,
            "q75": q75,
            "radius": radius[:, 0],
            "ll_mean": ll_mean[:, 0],
            "center_dist": center_dist[:, 0],
            "eff_modes": eff_modes[:, 0],
            "basin_mass": center_mass,
            "target_mass": target_mass,
            "target_min": target_min[:, 0],
            "target_margin": target_margin[:, 0],
        }

    def score_candidates(self, z: torch.Tensor, logw: torch.Tensor, candidate_z: torch.Tensor,
                         observed_hist: torch.Tensor | None = None,
                         target_freq: torch.Tensor | None = None,
                         target_centroids: torch.Tensor | None = None):
        state_ctx, state_stats = self.state_context(z, logw, observed_hist, target_freq)
        cand_feat, cand_stats = self.candidate_features(candidate_z, z, logw, state_stats["centers"], target_centroids)
        logits = self.query_policy(state_ctx, cand_feat)
        if observed_hist is not None and target_freq is not None:
            hist_norm = observed_hist / observed_hist.sum().clamp_min(1.0)
            deficit = (target_freq - hist_norm).clamp_min(0.0)
            cover_bonus = cand_stats["target_mass"] @ deficit
            logits = logits + 0.12 * cover_bonus
            cand_stats["coverage_bonus"] = cover_bonus
        return logits, state_ctx, state_stats, cand_feat, cand_stats

    def _basin_score(self, mass_t: torch.Tensor, centers_t: torch.Tensor, target_centroids: torch.Tensor, target_freq: torch.Tensor) -> torch.Tensor:
        d_forw = torch.cdist(centers_t, target_centroids).min(dim=1).values.mean()
        d_back = torch.cdist(target_centroids, centers_t).min(dim=1).values.mean()
        centroid_err = 0.5 * (d_forw + d_back)
        mass_err = torch.abs(torch.sort(mass_t)[0] - torch.sort(target_freq[:self.num_clusters])[0]).mean()
        return centroid_err + 0.75 * mass_err

    def teacher_action(self, z: torch.Tensor, logw: torch.Tensor, candidate_z_teacher: torch.Tensor,
                       candidate_y: torch.Tensor, target_centroids: torch.Tensor, target_freq: torch.Tensor,
                       observed_hist: torch.Tensor):
        scores = []
        teacher_clouds = []
        teacher_summaries = []
        teacher_drifts = []
        hist_norm = observed_hist / observed_hist.sum().clamp_min(1.0)
        deficit = (target_freq - hist_norm).clamp_min(0.0)
        missing = (hist_norm < 0.70 * target_freq).float()
        for j in range(candidate_z_teacher.shape[0]):
            zt, wt, drift = self.exact_teacher_update(z, logw, candidate_z_teacher[j])
            summ = hamiltonian_mode_summary(zt, wt, self.num_clusters, bandwidth=self.topology_bandwidth,
                                            merge_tol=self.mode_merge_tol, assign_tau=self.cluster_tau)
            mass_t, centers_t = summ["mass"], summ["centers"]
            label_freq = torch.bincount(candidate_y[j], minlength=self.num_classes).float()
            label_freq = label_freq / label_freq.sum().clamp_min(1.0)
            deficit_bonus = (deficit * label_freq).sum()
            missing_bonus = (missing * label_freq).sum()
            basin_pen = self._basin_score(mass_t, centers_t, target_centroids[:self.num_clusters], target_freq)
            eff_pen = 0.35 * (summ["effective_modes"] - min(self.num_clusters, self.num_classes)) ** 2
            query_spread = torch.pdist(candidate_z_teacher[j]).mean() if candidate_z_teacher[j].shape[0] > 1 else torch.tensor(0.0, device=z.device)
            score = 1.35 * deficit_bonus + 0.85 * missing_bonus - basin_pen - eff_pen - 0.015 * query_spread
            scores.append(score)
            teacher_clouds.append((zt, wt))
            teacher_summaries.append(summ)
            teacher_drifts.append(drift)
        scores_t = torch.stack(scores)
        best_idx = int(scores_t.argmax().item())
        sensitivity = (scores_t.max() - scores_t.min()).detach()
        return best_idx, scores_t.detach(), teacher_clouds, teacher_summaries, teacher_drifts, sensitivity

    def exact_teacher_update(self, z: torch.Tensor, logw: torch.Tensor, batch_z: torch.Tensor):
        drift, _, ll = self._score_field(z, batch_z, self.teacher_sigma)
        logw_t = logw + ll
        w = torch.softmax(logw_t, dim=0)
        idx_p = systematic_resample(w)
        zt = z[idx_p].detach().clone()
        for _ in range(int(self.teacher_move_steps)):
            drift_t, _, _ = self._score_field(zt, batch_z, self.teacher_sigma)
            zt = zt + self.teacher_move_step_size * drift_t + math.sqrt(2.0 * self.teacher_move_step_size) * self.rejuvenation * torch.randn_like(zt)
        wt = torch.full((self.num_particles,), 1.0 / self.num_particles, device=batch_z.device)
        return zt, wt, drift.detach()

    def _student_resample_move(self, z: torch.Tensor, p: torch.Tensor, logw: torch.Tensor, batch_z: torch.Tensor):
        ess = ess_from_logw(logw)
        if ess >= self.ess_resample_frac * self.num_particles:
            return z, p, logw, False, ess, ess
        w = torch.softmax(logw, dim=0)
        idx = systematic_resample(w)
        z_new = z[idx].detach().clone()
        p_new = 0.30 * p[idx].detach().clone()
        for _ in range(int(self.student_move_steps)):
            drift_t, _, _ = self._score_field(z_new, batch_z, self.obs_sigma)
            z_new = z_new + self.student_move_step_size * drift_t + math.sqrt(2.0 * self.student_move_step_size) * (0.70 * self.rejuvenation) * torch.randn_like(z_new)
        logw_new = torch.full_like(logw, -math.log(self.num_particles))
        ess_after = ess_from_logw(logw_new)
        return z_new, p_new, logw_new, True, ess, ess_after

    def forward_step(self, z: torch.Tensor, p: torch.Tensor, logw: torch.Tensor, h: torch.Tensor, batch_z: torch.Tensor):
        state_ctx, state_stats = self.state_context(z, logw)
        ctrl, h_new = self.policy(state_ctx, h)

        part_w = torch.softmax(logw, dim=0)
        obs_w = torch.full((batch_z.shape[0],), 1.0 / batch_z.shape[0], device=batch_z.device)
        support = torch.cat([z.detach(), batch_z.detach()], dim=0)
        support_w = torch.cat([
            (1.0 - self.obs_support_weight) * part_w,
            self.obs_support_weight * obs_w,
        ], dim=0)
        energy, gradU, level_val, learned_level, support_level = self._latent_level_energy(z, support, support_w, state_ctx)

        obs_drift, obs_target, _ = self._score_field(z, batch_z, self.obs_sigma)
        batch_summary = hamiltonian_mode_summary(batch_z, obs_w, min(self.num_clusters, batch_z.shape[0]),
                                                 bandwidth=self.topology_bandwidth, merge_tol=self.mode_merge_tol,
                                                 assign_tau=self.cluster_tau)
        batch_centers, batch_mass = batch_summary["centers"], batch_summary["mass"]
        topo_push = torch.zeros_like(z)
        for c in range(batch_centers.shape[0]):
            direction = batch_centers[c][None, :] - z
            topo_push = topo_push + batch_mass[c] * direction
        topo_push = topo_push / max(batch_centers.shape[0], 1)

        obs_term = ctrl["obs_gain"] * torch.tanh(obs_drift / 3.0)
        topo_term = ctrl["topo_gain"] * torch.tanh(topo_push / 4.0)
        force = -gradU + obs_term + topo_term
        p_new = (1.0 - ctrl["gamma"]) * p + ctrl["dt"] * force
        vel = self.kinetic.velocity(p_new)
        z_new = z + ctrl["dt"] * vel

        _, _, ll = self._score_field(z_new, batch_z, self.obs_sigma)
        logw_new = logw + ll
        logw_new = logw_new - torch.logsumexp(logw_new, dim=0)
        z_new, p_new, logw_new, resampled, ess_pre, ess_post = self._student_resample_move(z_new, p_new, logw_new, batch_z)
        post_summary = self._hamiltonian_summary(z_new.detach(), logw_new.detach())

        diag = {
            "energy": energy.detach(),
            "level_value": level_val.detach(),
            "learned_level": learned_level.detach(),
            "support_level": support_level.detach(),
            "state_ctx": state_ctx.detach(),
            "state_stats": state_stats,
            "post_summary": post_summary,
            "ctrl": ctrl,
            "u": (obs_term + topo_term).detach(),
            "net_drift": force.detach(),
            "obs_target": obs_target.detach(),
            "obs_drift": obs_drift.detach(),
            "batch_centers": batch_centers.detach(),
            "kinetic_energy": self.kinetic(p_new).detach(),
            "resampled": bool(resampled),
            "ess_pre": float(ess_pre.detach().cpu().item()),
            "ess_post": float(ess_post.detach().cpu().item()),
        }
        return z_new, p_new, logw_new, h_new, diag
