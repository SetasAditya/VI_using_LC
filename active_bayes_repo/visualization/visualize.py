from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import ensure_dir
from visualization.plots import (
    PALETTE,
    adjusted_rand_index,
    cluster_entropy,
    component_centroids_from_grid,
    connected_components_2d,
    contour_density_from_energy,
    effective_num_clusters,
    kde_grid_from_points,
    make_grid,
    normalized_mutual_info,
    purity_score,
    soft_assign_points,
    threshold_for_k_components,
)


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _history_stats(history: List[dict]) -> Dict[str, np.ndarray]:
    rows = {k: [] for k in [
        "purity", "ari", "nmi", "ess", "transport_gap", "query_acc", "sensitivity", "query_regret", "resampled",
        "dt", "gamma", "obs_gain", "topo_gain", "effective_modes",
    ]}
    rows["observed_hist"] = []
    rows["target_freq"] = []
    rows["selected_idx"] = []
    rows["teacher_idx"] = []
    rows["teacher_score_gap"] = []

    for item in history:
        batch_z = _np(item["batch_z_true"])
        batch_y = _np(item["batch_y"]).astype(int)
        centers = _np(item["centers"])
        pred = soft_assign_points(batch_z, centers, tau=0.7).argmax(axis=1)
        rows["purity"].append(purity_score(batch_y, pred))
        rows["ari"].append(adjusted_rand_index(batch_y, pred))
        rows["nmi"].append(normalized_mutual_info(batch_y, pred))
        w = torch.softmax(item["logw"], dim=0).detach().cpu().numpy()
        rows["ess"].append(float(1.0 / np.sum(w ** 2)))
        rows["transport_gap"].append(float(torch.norm(item["particles"].mean(0) - item["teacher_cloud"].mean(0)).item()))
        rows["query_acc"].append(float(item["selected_idx"] == item["teacher_idx"]))
        rows["sensitivity"].append(float(item["sensitivity"]))
        ts = _np(item["teacher_scores"])
        rows["query_regret"].append(float(np.max(ts) - ts[int(item["selected_idx"])]))
        rows["resampled"].append(float(item.get("resampled", False)))
        rows["dt"].append(float(item["ctrl"]["dt"]))
        rows["gamma"].append(float(item["ctrl"]["gamma"]))
        rows["obs_gain"].append(float(item["ctrl"]["obs_gain"]))
        rows["topo_gain"].append(float(item["ctrl"]["topo_gain"]))
        em = item.get("effective_modes", np.nan)
        rows["effective_modes"].append(float(_np(em)))
        rows["observed_hist"].append(_np(item["observed_hist"]))
        rows["target_freq"].append(_np(item["target_freq"]))
        rows["selected_idx"].append(int(item["selected_idx"]))
        rows["teacher_idx"].append(int(item["teacher_idx"]))
        rows["teacher_score_gap"].append(float(np.max(ts) - np.min(ts)))
    return {k: np.asarray(v) for k, v in rows.items()}


def _gmm_density_from_meta(X: np.ndarray, Y: np.ndarray, dataset_split: Dict | None = None):
    if dataset_split is None or not isinstance(dataset_split.get("meta", None), dict):
        return None
    meta = dataset_split["meta"]
    mu = np.asarray(meta.get("mu_true", None))
    Sigma = np.asarray(meta.get("Sigma_true", None))
    pi = np.asarray(meta.get("pi_true", None))
    if mu is None or Sigma is None or pi is None:
        return None
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dens = np.zeros((pts.shape[0],), dtype=float)
    for k in range(mu.shape[0]):
        Sk = Sigma[k] + 1e-6 * np.eye(Sigma.shape[-1])
        diff = pts - mu[k][None, :]
        inv = np.linalg.inv(Sk)
        quad = np.sum((diff @ inv) * diff, axis=1)
        norm = 1.0 / np.sqrt((2.0 * np.pi) ** pts.shape[1] * np.linalg.det(Sk))
        dens += float(pi[k]) * norm * np.exp(-0.5 * quad)
    return dens.reshape(X.shape)


def _student_density_from_model(assimilator, item: dict, X: np.ndarray, Y: np.ndarray, bandwidth: float):
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dev = next(assimilator.parameters()).device
    z = torch.as_tensor(pts, dtype=torch.float32, device=dev)
    ctx = torch.as_tensor(_np(item["state_ctx"]), dtype=torch.float32, device=dev)
    with torch.no_grad():
        dens = assimilator.learned_level(z, ctx).detach().cpu().numpy().reshape(X.shape)
    return dens


def _density_fields(assimilator, item: dict, X: np.ndarray, Y: np.ndarray, bandwidth: float, dataset_split: Dict | None = None):
    student = _student_density_from_model(assimilator, item, X, Y, bandwidth)
    true_d = _gmm_density_from_meta(X, Y, dataset_split)
    if true_d is None:
        batch = _np(item["batch_z_true"])
        wt = np.full((batch.shape[0],), 1.0 / max(batch.shape[0], 1), dtype=float)
        true_d = kde_grid_from_points(batch, wt, X, Y, bandwidth=bandwidth)
    return student, true_d


def render_distribution_panels(assimilator, history: List[dict], cfg, out_dir: str | Path, dataset_split: Dict | None = None):
    out_dir = ensure_dir(out_dir)
    steps = [min(int(s), len(history) - 1) for s in cfg.visualization.panel_steps]
    ncols = 3
    nrows = int(np.ceil(len(steps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
    axes = np.array(axes).reshape(-1)
    cmap = cm.get_cmap("plasma")
    norm = plt.Normalize(0, max(len(history) - 1, 1))

    xlim = tuple(getattr(cfg.visualization, "grid_lim_x", [-8, 8]))
    ylim = tuple(getattr(cfg.visualization, "grid_lim_y", [-8, 8]))
    X, Y, _ = make_grid(xlim, ylim, int(getattr(cfg.visualization, "grid_n", 160)))
    bw = float(getattr(cfg.model, "topology_bandwidth", 0.9))
    K = int(cfg.model.num_clusters)

    true_means = None
    if dataset_split is not None and isinstance(dataset_split.get("meta", {}), dict):
        true_means = np.asarray(dataset_split["meta"].get("mu_true", None)) if dataset_split.get("meta", {}) else None

    for ax in axes[len(steps):]:
        ax.axis("off")

    for ax, idx in zip(axes, steps):
        item = history[idx]
        student_d, true_d = _density_fields(assimilator, item, X, Y, bw, dataset_split)
        student_e = -np.log(student_d + 1e-8)
        ax.contourf(X, Y, student_d, levels=14, alpha=0.28, cmap="viridis")
        ax.contour(X, Y, student_e, levels=10, linewidths=0.85, cmap="viridis")

        thr_true, comps_true = threshold_for_k_components(true_d, K)
        thr_student, comps_student = threshold_for_k_components(student_d, K)
        ax.contour(X, Y, true_d, levels=[thr_true], colors="black", linewidths=1.6, linestyles="--")
        ax.contour(X, Y, student_d, levels=[thr_student], colors="#1f77b4", linewidths=1.4)
        ctrue, mtrue = component_centroids_from_grid(X, Y, true_d, comps_true)
        cstu, mstu = component_centroids_from_grid(X, Y, student_d, comps_student)
        if ctrue.shape[0]:
            ax.scatter(ctrue[:, 0], ctrue[:, 1], marker="x", s=95, color="black", linewidths=1.8, zorder=12)
        if cstu.shape[0]:
            ax.scatter(cstu[:, 0], cstu[:, 1], marker="*", s=220, color="white", edgecolors="black", linewidths=0.9, zorder=13)

        batch = _np(item["batch_z_true"])
        by = _np(item["batch_y"]).astype(int)
        for c in np.unique(by):
            mask = by == c
            ax.scatter(batch[mask, 0], batch[mask, 1], s=18, alpha=0.60, color=PALETTE[c % len(PALETTE)], edgecolors="none")
        particles = _np(item["particles"])
        w = np.exp(_np(item["logw"]) - np.max(_np(item["logw"])))
        w = w / (w.sum() + 1e-8)
        ax.scatter(particles[:, 0], particles[:, 1], s=10 + 45 * w, alpha=0.18, color="#5B84B1", edgecolors="none")
        if true_means is not None:
            ax.scatter(true_means[:, 0], true_means[:, 1], marker="+", s=110, color="black", linewidths=1.2, zorder=14)

        pur = purity_score(by, soft_assign_points(batch, _np(item["centers"]), tau=0.7).argmax(axis=1))
        ari = adjusted_rand_index(by, soft_assign_points(batch, _np(item["centers"]), tau=0.7).argmax(axis=1))
        ess = float(1.0 / np.sum(w ** 2))
        effm = float(_np(item.get("effective_modes", np.nan)))
        ax.text(0.02, 0.98, f"pur={pur:.2f} ari={ari:.2f}\ness={ess:.1f} Km={effm:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
        ax.set_title(f"t = {item['step']}", color=cmap(norm(idx)))
        ax.set_xlabel("latent-1")
        ax.set_ylabel("latent-2")
        ax.grid(alpha=0.15)

    fig.suptitle("Latent density capture: true mode topology vs student level-set topology", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(out_dir) / "active_distribution_evolution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_query_diagnostics(history: List[dict], out_dir: str | Path, stats: Dict[str, np.ndarray]):
    out_dir = ensure_dir(out_dir)
    T = np.arange(len(history))
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    ax = axes[0]
    ax.plot(T, stats["selected_idx"], "-o", label="selected", lw=2)
    ax.plot(T, stats["teacher_idx"], "-s", label="teacher", lw=2)
    ax.set_ylabel("action id")
    ax.set_title("Query policy vs Bayes teacher")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(T, stats["query_acc"], "-o", lw=2, label="query acc")
    ax.plot(T, stats["sensitivity"], "-o", lw=2, label="teacher sensitivity")
    ax.plot(T, stats["teacher_score_gap"], "-o", lw=2, label="score gap")
    ax.plot(T, stats["query_regret"], "-o", lw=2, label="query regret")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[2]
    logits = np.stack([_np(item["query_logits"]) for item in history], axis=0)
    im = ax.imshow(logits.T, aspect="auto", cmap="coolwarm")
    ax.set_ylabel("candidate id")
    ax.set_xlabel("step")
    ax.set_title("Query logits heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.tight_layout()
    out = Path(out_dir) / "query_policy_diagnostics.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_mode_transport(history: List[dict], out_dir: str | Path, stats: Dict[str, np.ndarray], cfg):
    out_dir = ensure_dir(out_dir)
    T = np.arange(len(history))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    ax = axes[0, 0]
    ax.plot(T, stats["purity"], "-o", lw=2, label="purity")
    ax.plot(T, stats["ari"], "-o", lw=2, label="ARI")
    ax.plot(T, stats["nmi"], "-o", lw=2, label="NMI")
    ax.plot(T, stats["effective_modes"] / max(int(cfg.model.num_clusters), 1), "-o", lw=2, label="eff_modes/K")
    ax.set_ylim(-0.05, 1.15)
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_title("Mode identification + topological identifiability")

    ax = axes[0, 1]
    ax.plot(T, stats["ess"], "-o", lw=2, label="ESS")
    ax.plot(T, stats["transport_gap"], "-o", lw=2, label="transport gap")
    ax.plot(T, stats["resampled"] * np.nanmax(stats["ess"]), "--", lw=1.5, label="resampled")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_title("Posterior health")

    ax = axes[1, 0]
    target = stats["target_freq"][0]
    obs = stats["observed_hist"]
    obs_norm = obs / np.maximum(obs.sum(axis=1, keepdims=True), 1.0)
    for c in range(obs_norm.shape[1]):
        ax.plot(T, obs_norm[:, c], "-o", lw=1.8, color=PALETTE[c % len(PALETTE)], label=f"obs label {c}")
        ax.axhline(target[c], color=PALETTE[c % len(PALETTE)], ls="--", lw=1.0)
    ax.grid(alpha=0.25)
    ax.set_title("Coverage of queried labels vs episode target")
    ax.legend(ncol=min(4, obs_norm.shape[1]))
    ax.set_xlabel("step")

    ax = axes[1, 1]
    for key in ["dt", "gamma", "obs_gain", "topo_gain"]:
        ax.plot(T, stats[key], "-o", lw=1.8, label=key)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    ax.set_title("Controller outputs")
    ax.set_xlabel("step")

    fig.tight_layout()
    out = Path(out_dir) / "mode_transport_diagnostics.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_final_alignment(history: List[dict], out_dir: str | Path, cfg, dataset_split: Dict | None = None):
    out_dir = ensure_dir(out_dir)
    item = history[-1]
    xlim = tuple(getattr(cfg.visualization, "grid_lim_x", [-8, 8]))
    ylim = tuple(getattr(cfg.visualization, "grid_lim_y", [-8, 8]))
    X, Y, _ = make_grid(xlim, ylim, int(getattr(cfg.visualization, "grid_n", 160)))
    bw = float(getattr(cfg.model, "topology_bandwidth", 0.9))
    K = int(cfg.model.num_clusters)
    student_d, true_d = _density_fields(assimilator, item, X, Y, bw, dataset_split)
    thr_true, comps_true = threshold_for_k_components(true_d, K)
    thr_student, comps_student = threshold_for_k_components(student_d, K)
    ctrue, _ = component_centroids_from_grid(X, Y, true_d, comps_true)
    cstu, _ = component_centroids_from_grid(X, Y, student_d, comps_student)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    # True density / topology
    axes[0].contourf(X, Y, true_d, levels=14, cmap="Greys", alpha=0.30)
    axes[0].contour(X, Y, true_d, levels=[thr_true], colors="black", linewidths=1.7, linestyles="--")
    if dataset_split is not None and "z" in dataset_split:
        z_true = np.asarray(dataset_split["z"])
        y_true = np.asarray(dataset_split["y"]).astype(int)
        for c in np.unique(y_true):
            mask = y_true == c
            axes[0].scatter(z_true[mask, 0], z_true[mask, 1], s=8, alpha=0.18, color=PALETTE[c % len(PALETTE)], edgecolors="none")
    if ctrue.shape[0]:
        axes[0].scatter(ctrue[:, 0], ctrue[:, 1], marker="x", s=90, color="black")
    axes[0].set_title("True data density and K-mode level set")

    # Student density / topology
    particles = _np(item["particles"])
    w = np.exp(_np(item["logw"]) - np.max(_np(item["logw"])))
    w = w / (w.sum() + 1e-8)
    axes[1].contourf(X, Y, student_d, levels=14, cmap="viridis", alpha=0.30)
    axes[1].contour(X, Y, student_d, levels=[thr_student], colors="#1f77b4", linewidths=1.7)
    axes[1].scatter(particles[:, 0], particles[:, 1], s=10 + 45 * w, alpha=0.18, color="#4C72B0", edgecolors="none")
    if cstu.shape[0]:
        axes[1].scatter(cstu[:, 0], cstu[:, 1], marker="*", s=220, color="white", edgecolors="black")
    axes[1].set_title("Student latent density and inferred basins")

    # Overlay centers
    if ctrue.shape[0]:
        axes[2].scatter(ctrue[:, 0], ctrue[:, 1], marker="x", s=100, color="black", label="true mode")
    if cstu.shape[0]:
        axes[2].scatter(cstu[:, 0], cstu[:, 1], marker="*", s=220, color="white", edgecolors="black", label="student basin")
    axes[2].set_title("True modes vs student basins")
    axes[2].legend()
    axes[2].grid(alpha=0.2)

    batch = _np(item["batch_z_true"])
    y = _np(item["batch_y"]).astype(int)
    centers = _np(item["centers"])
    for c in np.unique(y):
        mask = y == c
        axes[3].scatter(batch[mask, 0], batch[mask, 1], s=24, c=PALETTE[c % len(PALETTE)], alpha=0.7)
    axes[3].scatter(centers[:, 0], centers[:, 1], marker="*", s=220, color="white", edgecolors="black")
    axes[3].set_title("Queried batch vs inferred basins")
    axes[3].grid(alpha=0.2)

    for ax in axes:
        ax.set_xlabel("latent-1")
        ax.set_ylabel("latent-2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    fig.tight_layout()
    out = Path(out_dir) / "final_alignment.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out




def render_passive_distribution_panels(assimilator, history: List[dict], cfg, out_dir: str | Path, dataset_split: Dict | None = None):
    out_dir = ensure_dir(out_dir)
    steps = [min(int(s), len(history) - 1) for s in cfg.visualization.panel_steps]
    ncols = 3
    nrows = int(np.ceil(len(steps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
    axes = np.array(axes).reshape(-1)

    xlim = tuple(getattr(cfg.visualization, "grid_lim_x", [-8, 8]))
    ylim = tuple(getattr(cfg.visualization, "grid_lim_y", [-8, 8]))
    X, Y, _ = make_grid(xlim, ylim, int(min(getattr(cfg.visualization, "grid_n", 160), 120)))
    bw = float(getattr(cfg.model, "topology_bandwidth", 0.9))

    z_all = np.asarray(dataset_split["z"]) if dataset_split is not None and "z" in dataset_split else None
    y_all = np.asarray(dataset_split["y"]).astype(int) if dataset_split is not None and "y" in dataset_split else None
    target_centroids = np.asarray(dataset_split["target_centroids"]) if dataset_split is not None and "target_centroids" in dataset_split else None

    for ax in axes[len(steps):]:
        ax.axis("off")

    for ax, idx in zip(axes, steps):
        item = history[idx]
        particles = _np(item["particles"])
        logw = _np(item["logw"])
        w = np.exp(logw - np.max(logw)); w = w / (w.sum() + 1e-8)
        dens = _student_density_from_model(assimilator, item, X, Y, bw)
        ax.contourf(X, Y, dens, levels=12, cmap="viridis", alpha=0.28)
        ax.contour(X, Y, dens, levels=8, colors="#1f77b4", linewidths=1.0, alpha=0.8)

        if z_all is not None:
            for c in np.unique(y_all):
                mask = y_all == c
                ax.scatter(z_all[mask, 0], z_all[mask, 1], s=6, alpha=0.08, color=PALETTE[int(c) % len(PALETTE)], edgecolors="none")

        batch = _np(item["batch_z_true"])
        by = _np(item["batch_y"]).astype(int)
        for c in np.unique(by):
            mask = by == c
            ax.scatter(batch[mask, 0], batch[mask, 1], s=22, alpha=0.70, color=PALETTE[int(c) % len(PALETTE)], edgecolors="none")

        ax.scatter(particles[:, 0], particles[:, 1], s=10 + 35 * w, alpha=0.18, color="#4C72B0", edgecolors="none")
        centers = _np(item["centers"])
        ax.scatter(centers[:, 0], centers[:, 1], marker="*", s=220, color="white", edgecolors="black", linewidths=0.9, zorder=12)
        if target_centroids is not None:
            ax.scatter(target_centroids[:, 0], target_centroids[:, 1], marker="x", s=90, color="black", linewidths=1.5, zorder=13)

        pred = soft_assign_points(batch, centers, tau=0.7).argmax(axis=1)
        pur = purity_score(by, pred)
        ari = adjusted_rand_index(by, pred)
        ess = float(1.0 / np.sum(w ** 2))
        effm = float(_np(item.get("effective_modes", np.nan)))
        ax.text(0.02, 0.98, f"pur={pur:.2f} ari={ari:.2f}\ness={ess:.1f} Km={effm:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
        ax.set_title(f"t = {item['step']} (iid passive batch)")
        ax.set_xlabel("latent-1")
        ax.set_ylabel("latent-2")
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(alpha=0.15)

    fig.suptitle("Sequential passive assimilation: unbiased iid batches and latent mode capture", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(out_dir) / "active_distribution_evolution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_passive_final_alignment(history: List[dict], out_dir: str | Path, cfg, dataset_split: Dict | None = None):
    out_dir = ensure_dir(out_dir)
    item = history[-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    z_all = np.asarray(dataset_split["z"]) if dataset_split is not None and "z" in dataset_split else None
    y_all = np.asarray(dataset_split["y"]).astype(int) if dataset_split is not None and "y" in dataset_split else None
    target_centroids = np.asarray(dataset_split["target_centroids"]) if dataset_split is not None and "target_centroids" in dataset_split else None
    particles = _np(item["particles"])
    logw = _np(item["logw"]); w = np.exp(logw - np.max(logw)); w = w / (w.sum() + 1e-8)
    centers = _np(item["centers"])
    batch = _np(item["batch_z_true"])
    by = _np(item["batch_y"]).astype(int)
    teacher = _np(item["teacher_cloud"])

    if z_all is not None:
        for c in np.unique(y_all):
            mask = y_all == c
            axes[0].scatter(z_all[mask, 0], z_all[mask, 1], s=8, alpha=0.20, color=PALETTE[int(c) % len(PALETTE)], edgecolors="none")
    if target_centroids is not None:
        axes[0].scatter(target_centroids[:, 0], target_centroids[:, 1], marker="x", s=100, color="black", linewidths=1.5)
    axes[0].set_title("True latent data modes")

    axes[1].scatter(particles[:, 0], particles[:, 1], s=10 + 40 * w, alpha=0.20, color="#4C72B0", edgecolors="none", label="student")
    axes[1].scatter(teacher[:, 0], teacher[:, 1], s=10, alpha=0.12, color="#DD8452", edgecolors="none", label="teacher")
    axes[1].scatter(centers[:, 0], centers[:, 1], marker="*", s=220, color="white", edgecolors="black")
    axes[1].set_title("Student cloud, teacher cloud, inferred basins")
    axes[1].legend(fontsize=8)

    for c in np.unique(by):
        mask = by == c
        axes[2].scatter(batch[mask, 0], batch[mask, 1], s=24, alpha=0.75, color=PALETTE[int(c) % len(PALETTE)], edgecolors="none")
    axes[2].scatter(centers[:, 0], centers[:, 1], marker="*", s=220, color="white", edgecolors="black")
    if target_centroids is not None:
        axes[2].scatter(target_centroids[:, 0], target_centroids[:, 1], marker="x", s=100, color="black", linewidths=1.5)
    axes[2].set_title("Last unbiased batch vs inferred basins")

    xlim = tuple(getattr(cfg.visualization, "grid_lim_x", [-8, 8]))
    ylim = tuple(getattr(cfg.visualization, "grid_lim_y", [-8, 8]))
    for ax in axes:
        ax.set_xlabel("latent-1"); ax.set_ylabel("latent-2"); ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(alpha=0.2)
    fig.tight_layout()
    out = Path(out_dir) / "final_alignment.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_all_visualizations(assimilator, history: List[dict], cfg, out_dir: str | Path, dataset_split: Dict | None = None):
    out_dir = ensure_dir(out_dir)
    stats = _history_stats(history)
    passive = str(getattr(cfg.data, "query_scheme", "active_localized")) == "passive_iid"
    outputs = {
        "distribution": (render_passive_distribution_panels(assimilator, history, cfg, out_dir, dataset_split) if passive else render_distribution_panels(assimilator, history, cfg, out_dir, dataset_split)),
        "query_policy": render_query_diagnostics(history, out_dir, stats),
        "mode_transport": render_mode_transport(history, out_dir, stats, cfg),
        "final_alignment": (render_passive_final_alignment(history, out_dir, cfg, dataset_split) if passive else render_final_alignment(history, out_dir, cfg, dataset_split)),
    }
    summary = {
        "final_purity": float(stats["purity"][-1]),
        "final_ari": float(stats["ari"][-1]),
        "final_nmi": float(stats["nmi"][-1]),
        "final_ess": float(stats["ess"][-1]),
        "final_effective_modes": float(stats["effective_modes"][-1]),
        "query_accuracy": float(stats["query_acc"].mean()),
        "mean_transport_gap": float(stats["transport_gap"].mean()),
        "mean_sensitivity": float(stats["sensitivity"].mean()),
        "mean_query_regret": float(stats["query_regret"].mean()),
    }
    summary_path = Path(out_dir) / "visualization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    outputs["summary"] = summary_path
    return outputs
