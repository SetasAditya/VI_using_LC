from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.losses import active_assimilation_loss, gmm_logprob_torch, projector_loss, query_supervision_loss
from models.topology import hamiltonian_mode_summary
from models.utils import sinkhorn_divergence


def query_disabled(cfg) -> bool:
    return bool(getattr(cfg.training, "disable_query_training", False)) or str(getattr(cfg.data, "query_scheme", "active_localized")) == "passive_iid"


def make_loader(split: Dict[str, np.ndarray], batch_size: int = 128, shuffle: bool = True):
    ds = TensorDataset(
        torch.as_tensor(split["x"], dtype=torch.float32),
        torch.as_tensor(split["y"], dtype=torch.long),
        torch.as_tensor(split["z"], dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_projector(projector, cfg, dataset, device, output_dir: Path):
    loader = make_loader(dataset["train"], batch_size=128, shuffle=True)
    val_loader = make_loader(dataset["val"], batch_size=256, shuffle=False)
    opt = torch.optim.Adam(projector.parameters(), lr=cfg.training.projector_lr, weight_decay=cfg.training.weight_decay)
    best_acc, best_state = -1.0, None
    history = []

    for epoch in range(1, int(cfg.training.projector_epochs) + 1):
        projector.train()
        train_losses = []
        for x, y, z_true in loader:
            x, y, z_true = x.to(device), y.to(device), z_true.to(device)
            z, logits = projector(x)
            loss = projector_loss(z, logits, y, float(cfg.training.cls_weight), float(cfg.training.center_weight))
            loss = loss + float(getattr(cfg.training, "proj_recon_weight", 0.0)) * F.mse_loss(z, z_true)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), float(cfg.training.grad_clip))
            opt.step()
            train_losses.append(float(loss.item()))

        projector.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                _, logits = projector(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        acc = correct / max(total, 1)
        history.append({"epoch": epoch, "loss": float(np.mean(train_losses)), "val_acc": acc})
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in projector.state_dict().items()}
        if epoch % int(cfg.training.print_every) == 0 or epoch == 1:
            print(f"[projector] epoch={epoch:03d} loss={np.mean(train_losses):.4f} val_acc={acc:.4f}")

    projector.load_state_dict(best_state)
    return {"best_val_acc": best_acc, "history": history}


def _topology_parameter_groups(assimilator) -> Iterable[torch.nn.Parameter]:
    modules = [assimilator.level_head, assimilator.energy]
    for m in modules:
        for p in m.parameters():
            yield p


def _query_parameter_groups(assimilator) -> Iterable[torch.nn.Parameter]:
    modules = [assimilator.state_token_encoder, assimilator.coverage_proj, assimilator.candidate_proj, assimilator.query_policy]
    for m in modules:
        for p in m.parameters():
            yield p


def _to_torch(split: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in split.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in 'iu':
                out[k] = torch.as_tensor(v, dtype=torch.long, device=device)
            else:
                out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
        else:
            out[k] = v
    return out


def _candidate_label_freq(cand_y: torch.Tensor, num_classes: int) -> torch.Tensor:
    onehot = torch.nn.functional.one_hot(cand_y.long(), num_classes=num_classes).float()
    return onehot.mean(dim=1)


def _coverage_scores_from_labels(cand_y: torch.Tensor, observed_hist: torch.Tensor, target_freq: torch.Tensor,
                                 step_idx: int, total_steps: int) -> torch.Tensor:
    cand_freq = _candidate_label_freq(cand_y, target_freq.numel())
    hist_norm = observed_hist / observed_hist.sum().clamp_min(1.0)
    deficit = (target_freq - hist_norm).clamp_min(0.0)
    missing = (hist_norm < 0.70 * target_freq).float()
    rarity = 1.0 / (hist_norm + 1e-3)
    rarity = rarity / rarity.sum().clamp_min(1e-8)
    entropy_like = -(cand_freq * (cand_freq + 1e-8).log()).sum(dim=1)
    phase = 1.0 - float(step_idx) / max(int(total_steps) - 1, 1)
    score = (1.20 + 0.50 * phase) * (cand_freq @ deficit)
    score = score + (0.70 + 0.30 * phase) * (cand_freq @ missing)
    score = score + 0.10 * entropy_like + 0.10 * (cand_freq @ rarity)
    return score


def select_action(policy_mode: str, query_logits: torch.Tensor, teacher_idx: int, cand_y: torch.Tensor,
                  candidate_anchor: torch.Tensor, observed_hist: torch.Tensor, target_freq: torch.Tensor,
                  state_stats: Dict[str, torch.Tensor], step_idx: int, rng) -> int:
    C = cand_y.shape[0]
    if policy_mode in {"passive_iid", "no_query"} or C <= 1:
        return 0
    if policy_mode == "teacher":
        return int(teacher_idx)
    if policy_mode == "random":
        return int(rng.integers(0, C))
    if policy_mode == "round_robin":
        return int(step_idx % C)

    cand_freq = _candidate_label_freq(cand_y, target_freq.numel())
    hist_norm = observed_hist / observed_hist.sum().clamp_min(1.0)
    deficit = (target_freq - hist_norm).clamp_min(0.0)

    if policy_mode == "deficit":
        return int((cand_freq @ deficit).argmax().item())
    if policy_mode == "coverage_then_deficit":
        entropy_like = -(cand_freq * (cand_freq + 1e-8).log()).sum(dim=1)
        score = 0.8 * (cand_freq @ deficit) + 0.2 * entropy_like
        return int(score.argmax().item())
    if policy_mode == "frontier":
        centers = state_stats["centers"]
        d = torch.cdist(candidate_anchor, centers).min(dim=1).values
        return int(d.argmax().item())
    if policy_mode == "centroid_cycle":
        c = int(step_idx % max(target_freq.numel(), 1))
        target = state_stats.get("target_centroids", None)
        if target is None:
            return int(step_idx % C)
        d = torch.norm(candidate_anchor - target[c].unsqueeze(0), dim=1)
        return int(d.argmin().item())
    if policy_mode == "coverage_guarded":
        score = query_logits + 0.35 * _coverage_scores_from_labels(cand_y, observed_hist, target_freq, step_idx, C)
        return int(score.argmax().item())
    return int(query_logits.argmax().item())


def _component_alignment_loss(assimilator, meta: dict, ctx: torch.Tensor | None = None):
    centers, _, weights = assimilator.learned_component_params(ctx)
    mu = torch.as_tensor(meta["mu_true"], dtype=centers.dtype, device=centers.device)
    pi = torch.as_tensor(meta["pi_true"], dtype=centers.dtype, device=centers.device)
    ot = sinkhorn_divergence(centers, weights, mu, pi, epsilon=0.30, n_iters=35)
    entropy = -(weights * (weights + 1e-8).log()).sum()
    eff_modes = torch.exp(entropy)
    count_pen = (eff_modes - float(mu.shape[0])) ** 2
    return ot + 0.10 * count_pen


def pretrain_topology(projector, assimilator, cfg, dataset, device, output_dir: Path):
    epochs = int(getattr(cfg.training, "topology_pretrain_epochs", 0))
    if epochs <= 0 or not isinstance(dataset["train"].get("meta", None), dict):
        return {"enabled": False, "history": [], "reason": "epochs<=0 or no synthetic meta"}

    params = list(projector.parameters()) + list(_topology_parameter_groups(assimilator))
    opt = torch.optim.Adam(params, lr=float(getattr(cfg.training, "topology_pretrain_lr", cfg.training.assimilator_lr)),
                           weight_decay=float(cfg.training.weight_decay))
    loader = make_loader(dataset["train"], batch_size=160, shuffle=True)
    val_loader = make_loader(dataset["val"], batch_size=256, shuffle=False)
    best_val = float("inf")
    best_proj = None
    best_assim = None
    history = []

    train_meta = dataset["train"]["meta"]
    val_meta = dataset["val"]["meta"]
    zero_ctx = torch.zeros(assimilator.hidden_dim, device=device)

    def epoch_loss(loader, meta, train: bool):
        losses = []
        for x, y, z_true in loader:
            x, y, z_true = x.to(device), y.to(device), z_true.to(device)
            z_pred, logits = projector(x)
            cls = projector_loss(z_pred, logits, y, float(cfg.training.cls_weight), float(cfg.training.center_weight))
            recon = F.mse_loss(z_pred, z_true)
            log_pred = assimilator.learned_log_density(z_pred, zero_ctx)
            log_true = gmm_logprob_torch(z_true, meta)
            dens = F.mse_loss(log_pred, log_true)
            comp = _component_alignment_loss(assimilator, meta, zero_ctx)
            u_w = torch.full((z_pred.shape[0],), 1.0 / z_pred.shape[0], device=device)
            summary = hamiltonian_mode_summary(z_pred, u_w, assimilator.num_clusters,
                                               bandwidth=assimilator.topology_bandwidth,
                                               merge_tol=assimilator.mode_merge_tol,
                                               assign_tau=assimilator.cluster_tau)
            mu = torch.as_tensor(meta["mu_true"], dtype=z_pred.dtype, device=device)
            pi = torch.as_tensor(meta["pi_true"], dtype=z_pred.dtype, device=device)
            basin = sinkhorn_divergence(summary["centers"], summary["mass"], mu, pi, epsilon=0.30, n_iters=30)
            loss = (
                float(getattr(cfg.training, "topology_cls_weight", 0.35)) * cls
                + float(getattr(cfg.training, "topology_recon_weight", 1.0)) * recon
                + float(getattr(cfg.training, "topology_density_weight", 1.2)) * dens
                + float(getattr(cfg.training, "topology_component_weight", 0.9)) * comp
                + float(getattr(cfg.training, "topology_basin_weight", 0.6)) * basin
            )
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(cfg.training.grad_clip))
                opt.step()
            losses.append([float(loss.detach().cpu().item()), float(recon.detach().cpu().item()), float(dens.detach().cpu().item()), float(comp.detach().cpu().item()), float(basin.detach().cpu().item())])
        arr = np.asarray(losses)
        return {"loss": float(arr[:, 0].mean()), "recon": float(arr[:, 1].mean()), "dens": float(arr[:, 2].mean()), "comp": float(arr[:, 3].mean()), "basin": float(arr[:, 4].mean())}

    for epoch in range(1, epochs + 1):
        projector.train(); assimilator.train()
        tr = epoch_loss(loader, train_meta, train=True)
        projector.eval(); assimilator.eval()
        with torch.no_grad():
            va = epoch_loss(val_loader, val_meta, train=False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        if va["loss"] < best_val:
            best_val = va["loss"]
            best_proj = {k: v.detach().cpu() for k, v in projector.state_dict().items()}
            best_assim = {k: v.detach().cpu() for k, v in assimilator.state_dict().items()}
        if epoch % int(cfg.training.print_every) == 0 or epoch == 1:
            print(f"[topology] epoch={epoch:03d} train={tr['loss']:.4f} val={va['loss']:.4f} recon={tr['recon']:.4f} dens={tr['dens']:.4f} comp={tr['comp']:.4f}")

    if best_proj is not None:
        projector.load_state_dict(best_proj)
    if best_assim is not None:
        assimilator.load_state_dict(best_assim)
    return {"enabled": True, "best_val_loss": best_val, "history": history}


def run_query_pretrain_episode(projector, assimilator, split, cfg, device, rng, coverage_blend: float = 0.0):
    tb = _to_torch(split, device)
    projector.eval(); assimilator.eval()
    z, p, logw, h, observed_hist = assimilator.init_state(device)
    total_q_loss = 0.0
    total_acc = 0.0
    total_regret = 0.0
    steps = int(tb["candidate_x"].shape[0])

    with torch.no_grad():
        target_centroids = tb["target_centroids"].float()
        target_freq = tb["target_freq"].float()

    for t in range(steps):
        cand_x = tb["candidate_x"][t]
        cand_y = tb["candidate_y"][t]
        cand_z_true = tb["candidate_z"][t]
        C, B, D = cand_x.shape
        with torch.no_grad():
            flat_z, _ = projector(cand_x.reshape(C * B, D))
            cand_z = flat_z.reshape(C, B, -1)

        query_logits, state_ctx, state_stats, cand_feat, cand_stats = assimilator.score_candidates(
            z, logw, cand_z, observed_hist=observed_hist, target_freq=target_freq, target_centroids=target_centroids
        )
        teacher_idx, teacher_scores, teacher_clouds, teacher_summaries, teacher_drifts, sensitivity = assimilator.teacher_action(
            z, logw, cand_z_true, cand_y, target_centroids, target_freq, observed_hist
        )
        coverage_scores = _coverage_scores_from_labels(cand_y, observed_hist, target_freq, t, steps)
        q_loss = query_supervision_loss(
            query_logits, teacher_scores, coverage_scores=coverage_scores,
            coverage_weight=float(coverage_blend),
            temperature=float(getattr(cfg.training, "teacher_score_temperature", 0.35)),
        )
        total_q_loss = total_q_loss + q_loss
        total_acc += float(int(query_logits.argmax().item()) == teacher_idx)
        total_regret += float(torch.max(teacher_scores).item() - teacher_scores[int(query_logits.argmax().item())].item())

        batch_y = cand_y[teacher_idx]
        observed_hist = observed_hist + torch.bincount(batch_y, minlength=assimilator.num_classes).float()

        with torch.no_grad():
            rollout = str(getattr(cfg.training, "query_pretrain_rollout", "teacher_cloud"))
            if rollout == "teacher_cloud":
                z = teacher_clouds[teacher_idx][0].detach().clone()
                p = torch.zeros_like(z)
                logw = torch.full_like(logw, -math.log(assimilator.num_particles))
            else:
                batch_z = cand_z[teacher_idx]
                z, p, logw, h, _ = assimilator.forward_step(z, p, logw, h, batch_z)

    denom = max(steps, 1)
    return total_q_loss / denom, {"query_acc": total_acc / denom, "query_regret": total_regret / denom}


def pretrain_query_policy(projector, assimilator, cfg, dataset, device, output_dir: Path):
    epochs = int(getattr(cfg.training, "query_pretrain_epochs", 0))
    if epochs <= 0 or query_disabled(cfg):
        return {"enabled": False, "history": [], "reason": "query disabled" if query_disabled(cfg) else "epochs<=0"}
    params = list(_query_parameter_groups(assimilator))
    opt = torch.optim.Adam(params, lr=float(getattr(cfg.training, "query_pretrain_lr", cfg.training.assimilator_lr)),
                           weight_decay=float(cfg.training.weight_decay))
    rng = np.random.default_rng(int(cfg.seed) + 321)
    best_val = float("inf")
    best_state = None
    history = []
    cover_start = float(getattr(cfg.training, "query_cover_start", 0.70))
    cover_end = float(getattr(cfg.training, "query_cover_end", 0.20))

    for epoch in range(1, epochs + 1):
        blend = cover_start + (cover_end - cover_start) * ((epoch - 1) / max(epochs - 1, 1))
        assimilator.train()
        opt.zero_grad()
        train_loss, train_stats = run_query_pretrain_episode(projector, assimilator, dataset["train"], cfg, device, rng, coverage_blend=blend)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(cfg.training.grad_clip))
        opt.step()
        assimilator.eval()
        with torch.no_grad():
            val_loss, val_stats = run_query_pretrain_episode(projector, assimilator, dataset["val"], cfg, device, rng, coverage_blend=blend)
        t_loss = float(train_loss.detach().cpu().item())
        v_loss = float(val_loss.detach().cpu().item())
        row = {"epoch": epoch, "coverage_blend": blend, "train_loss": t_loss, "val_loss": v_loss,
               "train_query_acc": train_stats["query_acc"], "val_query_acc": val_stats["query_acc"],
               "train_query_regret": train_stats["query_regret"], "val_query_regret": val_stats["query_regret"]}
        history.append(row)
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu() for k, v in assimilator.state_dict().items()}
        if epoch % int(cfg.training.print_every) == 0 or epoch == 1:
            print(f"[query-pre] epoch={epoch:03d} blend={blend:.2f} train={t_loss:.4f} val={v_loss:.4f} qacc={train_stats['query_acc']:.3f} regret={train_stats['query_regret']:.3f}")
    if best_state is not None:
        assimilator.load_state_dict(best_state)
    return {"enabled": True, "best_val_loss": best_val, "history": history}


def run_active_episode(projector, assimilator, split, cfg, device, rng, track_history: bool = False,
                       training: bool = False, teacher_forcing_prob: float | None = None,
                       policy_mode: str = "learned", coverage_guidance_prob: float = 0.0):
    tb = _to_torch(split, device)
    projector.eval()
    z, p, logw, h, observed_hist = assimilator.init_state(device)
    total_loss = 0.0
    parts_accum = {"query": 0.0, "cloud_ot": 0.0, "basin_ot": 0.0, "ident": 0.0, "drift": 0.0, "nll": 0.0, "contrastive": 0.0, "level_match": 0.0}
    history = []

    if teacher_forcing_prob is None:
        teacher_forcing_prob = float(cfg.training.teacher_forcing_prob)

    with torch.no_grad():
        target_centroids = tb["target_centroids"].float()
        target_freq = tb["target_freq"].float()
    gmm_meta = split.get("meta", None) if isinstance(split.get("meta", None), dict) else None

    for t in range(tb["candidate_x"].shape[0]):
        cand_x = tb["candidate_x"][t]
        cand_y = tb["candidate_y"][t]
        cand_z_true = tb["candidate_z"][t]
        cand_anchor = tb["candidate_anchor"][t]
        C, B, D = cand_x.shape
        with torch.no_grad():
            flat_z, _ = projector(cand_x.reshape(C * B, D))
            cand_z = flat_z.reshape(C, B, -1)

        query_logits, state_ctx, state_stats, cand_feat, cand_stats = assimilator.score_candidates(
            z, logw, cand_z, observed_hist=observed_hist, target_freq=target_freq, target_centroids=target_centroids
        )
        teacher_idx, teacher_scores, teacher_clouds, teacher_summaries, teacher_drifts, sensitivity = assimilator.teacher_action(
            z, logw, cand_z_true, cand_y, target_centroids, target_freq, observed_hist
        )
        coverage_scores = _coverage_scores_from_labels(cand_y, observed_hist, target_freq, t, tb["candidate_x"].shape[0])

        guidance_used = "none"
        if training and rng.random() < teacher_forcing_prob:
            selected_idx = teacher_idx
            guidance_used = "teacher"
        elif training and policy_mode == "learned" and rng.random() < float(coverage_guidance_prob):
            tmp_stats = dict(state_stats); tmp_stats["target_centroids"] = target_centroids
            selected_idx = select_action("centroid_cycle", query_logits, teacher_idx, cand_y, cand_anchor, observed_hist, target_freq, tmp_stats, t, rng)
            guidance_used = "centroid_cycle"
        else:
            tmp_stats = dict(state_stats); tmp_stats["target_centroids"] = target_centroids
            selected_idx = select_action(policy_mode, query_logits, teacher_idx, cand_y, cand_anchor, observed_hist, target_freq, tmp_stats, t, rng)

        batch_z = cand_z[selected_idx]
        batch_y = cand_y[selected_idx]
        batch_z_true = cand_z_true[selected_idx]
        teacher_cloud = teacher_clouds[teacher_idx]
        teacher_summary = teacher_summaries[teacher_idx]
        teacher_drift = teacher_drifts[teacher_idx]

        h_old = h.clone()
        z, p, logw, h, diag = assimilator.forward_step(z, p, logw, h, batch_z)
        loss, parts = active_assimilation_loss(
            assimilator, z, logw, h_old, h, batch_z, query_logits, teacher_scores,
            teacher_cloud, teacher_summary, teacher_drift, diag, target_centroids, target_freq, cfg,
            coverage_scores=coverage_scores, gmm_meta=gmm_meta,
        )
        total_loss = total_loss + loss
        for k, v in parts.items():
            parts_accum[k] += v
        observed_hist = observed_hist + torch.bincount(batch_y, minlength=assimilator.num_classes).float()

        if track_history:
            history.append({
                "step": t,
                "policy_mode": policy_mode,
                "guidance_used": guidance_used,
                "selected_idx": selected_idx,
                "teacher_idx": teacher_idx,
                "teacher_scores": teacher_scores.detach().cpu(),
                "coverage_scores": coverage_scores.detach().cpu(),
                "query_logits": query_logits.detach().cpu(),
                "sensitivity": float(sensitivity.cpu().item()),
                "candidate_anchor": cand_anchor.detach().cpu(),
                "candidate_y": cand_y.detach().cpu(),
                "candidate_z_true": cand_z_true.detach().cpu(),
                "candidate_z": cand_z.detach().cpu(),
                "batch_y": batch_y.detach().cpu(),
                "batch_z": batch_z.detach().cpu(),
                "batch_z_true": batch_z_true.detach().cpu(),
                "particles": z.detach().cpu(),
                "momenta": p.detach().cpu(),
                "logw": logw.detach().cpu(),
                "mass": diag["post_summary"]["mass"].detach().cpu(),
                "centers": diag["post_summary"]["centers"].detach().cpu(),
                "assignments": diag["post_summary"]["assignments"].detach().cpu(),
                "peak": diag["post_summary"]["peak"].detach().cpu(),
                "effective_modes": diag["post_summary"]["effective_modes"].detach().cpu(),
                "energy": diag["energy"].detach().cpu(),
                "state_ctx": diag["state_ctx"].detach().cpu(),
                "u": diag["u"].detach().cpu(),
                "net_drift": diag["net_drift"].detach().cpu(),
                "teacher_drift": teacher_drift.detach().cpu(),
                "obs_target": diag["obs_target"].detach().cpu(),
                "ctrl": {k: float(v.detach().cpu().item()) for k, v in diag["ctrl"].items()},
                "teacher_cloud": teacher_cloud[0].detach().cpu(),
                "teacher_cloud_w": teacher_cloud[1].detach().cpu(),
                "teacher_summary_mass": teacher_summary["mass"].detach().cpu(),
                "teacher_summary_centers": teacher_summary["centers"].detach().cpu(),
                "observed_hist": observed_hist.detach().cpu(),
                "target_freq": target_freq.detach().cpu(),
                "target_centroids": target_centroids.detach().cpu(),
                "resampled": bool(diag["resampled"]),
                "ess_pre": float(diag["ess_pre"]),
                "ess_post": float(diag["ess_post"]),
            })

    mean_loss = total_loss / max(int(tb["candidate_x"].shape[0]), 1)
    mean_parts = {k: v / max(int(tb["candidate_x"].shape[0]), 1) for k, v in parts_accum.items()}
    return mean_loss, mean_parts, history


def train_assimilator(projector, assimilator, cfg, dataset, device, output_dir: Path):
    topology_pretrain_stats = pretrain_topology(projector, assimilator, cfg, dataset, device, output_dir)
    query_pretrain_stats = pretrain_query_policy(projector, assimilator, cfg, dataset, device, output_dir)
    passive_mode = query_disabled(cfg)

    opt = torch.optim.Adam(assimilator.parameters(), lr=cfg.training.assimilator_lr, weight_decay=cfg.training.weight_decay)
    rng = np.random.default_rng(int(cfg.seed) + 123)
    history = []
    best_val, best_state = float("inf"), None
    total_epochs = int(cfg.training.assimilator_epochs)
    tf_start = float(getattr(cfg.training, "teacher_forcing_prob_start", cfg.training.teacher_forcing_prob))
    tf_end = float(getattr(cfg.training, "teacher_forcing_prob_end", cfg.training.teacher_forcing_prob))
    cg_start = float(getattr(cfg.training, "coverage_guidance_prob_start", 0.35))
    cg_end = float(getattr(cfg.training, "coverage_guidance_prob_end", 0.05))

    for epoch in range(1, total_epochs + 1):
        ratio = ((epoch - 1) / max(total_epochs - 1, 1))
        teacher_forcing = tf_start + (tf_end - tf_start) * ratio
        coverage_guidance = cg_start + (cg_end - cg_start) * ratio
        assimilator.train()
        opt.zero_grad()
        train_loss, train_parts, _ = run_active_episode(
            projector, assimilator, dataset["train"], cfg, device, rng,
            training=not passive_mode,
            teacher_forcing_prob=(0.0 if passive_mode else teacher_forcing),
            policy_mode=("passive_iid" if passive_mode else "learned"),
            coverage_guidance_prob=(0.0 if passive_mode else coverage_guidance),
        )
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(assimilator.parameters(), float(cfg.training.grad_clip))
        opt.step()

        assimilator.eval()
        val_loss, val_parts, _ = run_active_episode(
            projector, assimilator, dataset["val"], cfg, device, rng,
            training=False, teacher_forcing_prob=0.0,
            policy_mode=("passive_iid" if passive_mode else "learned"),
        )

        t_loss = float(train_loss.detach().cpu().item())
        v_loss = float(val_loss.detach().cpu().item())
        row = {
            "epoch": epoch,
            "train_loss": t_loss,
            "val_loss": v_loss,
            "teacher_forcing": teacher_forcing,
            "coverage_guidance": coverage_guidance,
            **{f"train_{k}": v for k, v in train_parts.items()},
            **{f"val_{k}": v for k, v in val_parts.items()},
        }
        history.append(row)
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu() for k, v in assimilator.state_dict().items()}
        if epoch % int(cfg.training.print_every) == 0 or epoch == 1:
            print(
                f"[assim] epoch={epoch:03d} tf={teacher_forcing:.2f} cov={coverage_guidance:.2f} "
                f"train={t_loss:.4f} val={v_loss:.4f} ident={train_parts['ident']:.3f} "
                f"ot={train_parts['cloud_ot']:.3f} drift={train_parts['drift']:.3f} level={train_parts['level_match']:.3f}"
            )

    assimilator.load_state_dict(best_state)
    return {"best_val_loss": best_val, "topology_pretrain": topology_pretrain_stats, "query_pretrain": query_pretrain_stats, "history": history}
