from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ensure_dir
from evaluation.metrics import summarize_history
from training.trainer import run_active_episode


def _render_policy_ablation(metrics_by_policy: dict, out_path: str | Path):
    out_path = Path(out_path)
    policies = list(metrics_by_policy.keys())
    keys = ["query_accuracy", "mean_query_regret", "mean_ari", "mean_purity", "mean_effective_modes", "mean_transport_gap"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.reshape(-1)
    for ax, key in zip(axes, keys):
        vals = [metrics_by_policy[p][key] for p in policies]
        ax.bar(np.arange(len(policies)), vals)
        ax.set_xticks(np.arange(len(policies)))
        ax.set_xticklabels(policies, rotation=25, ha="right")
        ax.set_title(key)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _render_policy_sanity(metrics_by_policy: dict, out_path: str | Path):
    keys = [p for p in ["learned", "round_robin", "centroid_cycle", "teacher"] if p in metrics_by_policy]
    out_path = Path(out_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.bar(np.arange(len(keys)), [metrics_by_policy[k]["mean_effective_modes"] for k in keys])
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=20)
    ax.axhline(4.0, ls="--", color="black", lw=1)
    ax.set_title("Mean effective modes")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.bar(np.arange(len(keys)), [metrics_by_policy[k]["mean_transport_gap"] for k in keys])
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=20)
    ax.set_title("Mean transport gap")
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    width = 0.18
    for j, k in enumerate(keys):
        vals = metrics_by_policy[k].get("final_observed_freq", [])
        ax.bar(np.arange(len(vals)) + (j - len(keys)/2) * width + width/2, vals, width=width, label=k)
    target = metrics_by_policy[keys[0]].get("target_freq", []) if keys else []
    for j, tv in enumerate(target):
        ax.axhline(tv, color=f"C{j}", ls="--", lw=0.8, alpha=0.5)
    ax.set_title("Final observed label frequencies")
    ax.set_xlabel("label id")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.bar(np.arange(len(keys)), [metrics_by_policy[k]["query_accuracy"] for k in keys])
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=20)
    ax.set_title("Query accuracy")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def evaluate_checkpoint(projector, assimilator, dataset, cfg, device, split_name: str, out_dir: str | Path):
    out_dir = ensure_dir(out_dir)
    rng = np.random.default_rng(int(cfg.seed) + 999)
    projector.eval()
    assimilator.eval()

    base_policy = "passive_iid" if str(getattr(cfg.data, "query_scheme", "active_localized")) == "passive_iid" else "learned"
    _, _, history = run_active_episode(projector, assimilator, dataset[split_name], cfg, device, rng, track_history=True, training=False, policy_mode=base_policy)
    metrics = summarize_history(history, sigma=float(cfg.model.obs_sigma))
    path = Path(out_dir) / f"metrics_{split_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    default_policies = ["passive_iid"] if str(getattr(cfg.data, "query_scheme", "active_localized")) == "passive_iid" else ["learned", "teacher", "centroid_cycle", "deficit", "round_robin", "random"]
    policies = list(getattr(getattr(cfg, "evaluation", {}), "ablation_policies", default_policies))
    ablation = {}
    histories = {base_policy: history}
    for i, policy in enumerate(policies):
        if str(policy) == base_policy:
            ablation[str(policy)] = metrics
            continue
        prng = np.random.default_rng(int(cfg.seed) + 1000 + 17 * i)
        _, _, hist_pol = run_active_episode(projector, assimilator, dataset[split_name], cfg, device, prng, track_history=True, training=False, policy_mode=str(policy))
        histories[str(policy)] = hist_pol
        ablation[str(policy)] = summarize_history(hist_pol, sigma=float(cfg.model.obs_sigma))
    ablation_path = Path(out_dir) / f"policy_ablation_{split_name}.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2)
    ablation_fig = _render_policy_ablation(ablation, Path(out_dir) / f"policy_ablation_{split_name}.png")
    sanity_fig = _render_policy_sanity(ablation, Path(out_dir) / f"policy_sanity_{split_name}.png")

    # Incorporate the fixed-policy sanity-check rollouts directly in evaluation outputs.
    from visualization.visualize import render_distribution_panels, render_mode_transport, render_final_alignment, _history_stats
    sanity_paths = {}
    sanity_policies = ["round_robin", "centroid_cycle"] if base_policy != "passive_iid" else []
    for pol in sanity_policies:
        if pol in histories:
            sdir = ensure_dir(Path(out_dir) / f"sanity_{pol}")
            stats = _history_stats(histories[pol])
            sanity_paths[f"{pol}_distribution"] = render_distribution_panels(assimilator, histories[pol], cfg, sdir, dataset[split_name])
            sanity_paths[f"{pol}_diagnostics"] = render_mode_transport(histories[pol], sdir, stats, cfg)
            sanity_paths[f"{pol}_alignment"] = render_final_alignment(histories[pol], sdir, cfg, dataset[split_name])

    return metrics, history, path, ablation, ablation_path, ablation_fig, sanity_fig, sanity_paths
