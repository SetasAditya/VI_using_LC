from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .topology import hamiltonian_mode_summary
from .utils import gaussian_kde_logprob, sinkhorn_divergence


def gmm_logprob_torch(z: torch.Tensor, gmm_meta: dict) -> torch.Tensor:
    mu = gmm_meta["mu_true"]
    Sigma = gmm_meta["Sigma_true"]
    pi = gmm_meta["pi_true"]
    if not torch.is_tensor(mu):
        device = z.device
        mu = torch.as_tensor(mu, dtype=z.dtype, device=device)
        Sigma = torch.as_tensor(Sigma, dtype=z.dtype, device=device)
        pi = torch.as_tensor(pi, dtype=z.dtype, device=device)
    eye = 1e-5 * torch.eye(Sigma.shape[-1], device=z.device, dtype=z.dtype)[None, :, :]
    Sigma = Sigma + eye
    inv = torch.linalg.inv(Sigma)
    diff = z[:, None, :] - mu[None, :, :]
    mahal = torch.einsum("nkd,kde,nke->nk", diff, inv, diff)
    logdet = torch.logdet(Sigma)
    d = z.shape[-1]
    logcomp = torch.log(pi.clamp_min(1e-8))[None, :] - 0.5 * (mahal + logdet[None, :] + d * math.log(2.0 * math.pi))
    return torch.logsumexp(logcomp, dim=1)


def projector_loss(z: torch.Tensor, logits: torch.Tensor, y: torch.Tensor, cls_weight: float = 1.0,
                   center_weight: float = 0.05):
    ce = F.cross_entropy(logits, y)
    centers = []
    compact = 0.0
    for c in torch.unique(y):
        mask = y == c
        mu = z[mask].mean(dim=0)
        centers.append(mu)
        compact = compact + ((z[mask] - mu) ** 2).sum(dim=1).mean()
    sep = torch.pdist(torch.stack(centers)).mean() if len(centers) > 1 else torch.tensor(0.0, device=z.device)
    return cls_weight * ce + center_weight * (compact - 0.25 * sep)


def query_supervision_loss(query_logits: torch.Tensor, teacher_scores: torch.Tensor,
                           coverage_scores: torch.Tensor | None = None,
                           temperature: float = 0.35,
                           coverage_weight: float = 0.0):
    teacher_probs = torch.softmax(teacher_scores / float(temperature), dim=0)
    target_probs = teacher_probs
    if coverage_scores is not None and float(coverage_weight) > 0.0:
        cover_probs = torch.softmax(coverage_scores / float(temperature), dim=0)
        target_probs = (1.0 - float(coverage_weight)) * teacher_probs + float(coverage_weight) * cover_probs
        target_probs = target_probs / target_probs.sum().clamp_min(1e-8)
    student_log_probs = torch.log_softmax(query_logits / float(temperature), dim=0)
    query_kl = F.kl_div(student_log_probs, target_probs, reduction="batchmean", log_target=False)
    target_idx = torch.tensor([int(torch.argmax(target_probs).item())], device=query_logits.device)
    query_ce = F.cross_entropy(query_logits.unsqueeze(0), target_idx)
    entropy = -(torch.softmax(query_logits, dim=0) * torch.log_softmax(query_logits, dim=0)).sum()
    return 0.55 * query_kl + 0.45 * query_ce - 0.01 * entropy



def topology_identifiability_loss(student_mass: torch.Tensor, student_centers: torch.Tensor, student_peak: torch.Tensor,
                                  target_centroids: torch.Tensor, target_freq: torch.Tensor, num_modes: int):
    basin_target = sinkhorn_divergence(
        student_centers, student_mass,
        target_centroids[:num_modes], target_freq[:num_modes],
        epsilon=0.30, n_iters=35,
    )
    entropy = -(student_mass * (student_mass + 1e-8).log()).sum()
    eff_modes = torch.exp(entropy)
    mode_count_pen = (eff_modes - float(num_modes)) ** 2
    sep = torch.cdist(student_centers, student_centers)
    sep = sep + torch.eye(student_centers.shape[0], device=student_centers.device) * 1e6
    sep_pen = torch.exp(-0.40 * sep.min(dim=1).values).mean()
    peak_pen = -student_peak.mean()
    return basin_target + 0.15 * mode_count_pen + 0.10 * sep_pen + 0.05 * peak_pen



def active_assimilation_loss(model, z_new: torch.Tensor, logw_new: torch.Tensor, h_old: torch.Tensor, h_new: torch.Tensor,
                             batch_z: torch.Tensor, query_logits: torch.Tensor, teacher_scores: torch.Tensor,
                             teacher_cloud: tuple[torch.Tensor, torch.Tensor], teacher_summary: dict,
                             teacher_drift: torch.Tensor, diag: dict, target_centroids: torch.Tensor,
                             target_freq: torch.Tensor, cfg, coverage_scores: torch.Tensor | None = None,
                             gmm_meta: dict | None = None):
    probs_new = torch.softmax(logw_new, dim=0)
    z_teacher, w_teacher = teacher_cloud

    post_summary = diag.get("post_summary")
    if post_summary is None:
        post_summary = hamiltonian_mode_summary(
            z_new, probs_new, model.num_clusters,
            bandwidth=model.topology_bandwidth,
            merge_tol=model.mode_merge_tol,
            assign_tau=model.cluster_tau,
        )

    mass_s = post_summary["mass"]
    centers_s = post_summary["centers"]
    peak_s = post_summary["peak"]

    teacher_mass = teacher_summary["mass"]
    teacher_centers = teacher_summary["centers"]

    query_loss = query_supervision_loss(
        query_logits,
        teacher_scores,
        coverage_scores=coverage_scores,
        coverage_weight=float(getattr(cfg.training, "query_coverage_weight", 0.0)),
        temperature=float(getattr(cfg.training, "teacher_score_temperature", 0.35)),
    )

    cloud_ot = sinkhorn_divergence(
        z_new, probs_new,
        z_teacher, w_teacher,
        epsilon=float(getattr(cfg.training, "sinkhorn_epsilon", 0.35)),
        n_iters=int(getattr(cfg.training, "sinkhorn_iters", 40)),
    )
    basin_ot = sinkhorn_divergence(
        centers_s, mass_s,
        teacher_centers, teacher_mass,
        epsilon=float(getattr(cfg.training, "basin_sinkhorn_epsilon", 0.45)),
        n_iters=int(getattr(cfg.training, "sinkhorn_iters", 40)),
    )
    ident = topology_identifiability_loss(mass_s, centers_s, peak_s, target_centroids, target_freq, model.num_clusters)

    pred_nll = -gaussian_kde_logprob(z_new, batch_z, logw_new, sigma=model.obs_sigma).mean()

    td = torch.tanh(teacher_drift / 3.0)
    sd = torch.tanh(diag["net_drift"] / 3.0)
    drift_loss = F.mse_loss(sd, td)

    pos_energy = model.learned_log_density(z_teacher, diag["state_ctx"]).mean()
    neg = 0.5 * torch.randn_like(z_teacher) + z_teacher[torch.randperm(z_teacher.shape[0])]
    neg_energy = model.learned_log_density(neg, diag["state_ctx"]).mean()
    contrastive = F.softplus(neg_energy - pos_energy)

    level_match = torch.tensor(0.0, device=z_new.device)
    if gmm_meta is not None:
        pred_logp = model.learned_log_density(batch_z, diag["state_ctx"])
        true_logp = gmm_logprob_torch(batch_z, gmm_meta)
        level_match = F.mse_loss(pred_logp, true_logp)

    action = (diag["u"] ** 2).sum(dim=1).mean()
    mem = ((h_new - h_old) ** 2).mean()

    loss = (
        float(cfg.training.query_weight) * query_loss
        + float(cfg.training.teacher_weight) * cloud_ot
        + float(cfg.training.basin_weight) * basin_ot
        + float(getattr(cfg.training, "identifiability_weight", 1.0)) * ident
        + float(cfg.training.drift_weight) * drift_loss
        + float(cfg.training.nll_weight) * pred_nll
        + float(cfg.training.contrastive_weight) * contrastive
        + float(getattr(cfg.training, "level_match_weight", 0.0)) * level_match
        + float(cfg.training.control_weight) * action
        + float(cfg.training.memory_weight) * mem
    )
    parts = {
        "query": float(query_loss.detach().cpu().item()),
        "cloud_ot": float(cloud_ot.detach().cpu().item()),
        "basin_ot": float(basin_ot.detach().cpu().item()),
        "ident": float(ident.detach().cpu().item()),
        "drift": float(drift_loss.detach().cpu().item()),
        "nll": float(pred_nll.detach().cpu().item()),
        "contrastive": float(contrastive.detach().cpu().item()),
        "level_match": float(level_match.detach().cpu().item()),
    }
    return loss, parts
