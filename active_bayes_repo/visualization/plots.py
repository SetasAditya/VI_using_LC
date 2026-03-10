from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def make_grid(xlim: Tuple[float, float], ylim: Tuple[float, float], n: int):
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    return X, Y, pts


def contour_density_from_energy(energy: np.ndarray) -> np.ndarray:
    density = np.exp(-(energy - np.nanmin(energy)))
    return density / (np.nanmax(density) + 1e-8)


def normalized_weights(logw: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(logw, np.ndarray):
        lw = logw - np.max(logw)
        w = np.exp(lw)
        return w / (w.sum() + 1e-8)
    w = torch.softmax(logw, dim=0)
    return w.detach().cpu().numpy()


def soft_assign_points(points: np.ndarray, centers: np.ndarray, tau: float = 0.8) -> np.ndarray:
    d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    logits = -d2 / max(tau, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / (probs.sum(axis=1, keepdims=True) + 1e-8)


def contingency_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_true: int | None = None, n_pred: int | None = None) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_true is None:
        n_true = int(y_true.max()) + 1 if y_true.size else 0
    if n_pred is None:
        n_pred = int(y_pred.max()) + 1 if y_pred.size else 0
    mat = np.zeros((n_true, n_pred), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if yt >= 0 and yp >= 0:
            mat[yt, yp] += 1
    return mat


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mat = contingency_matrix(y_true, y_pred)
    total = mat.sum()
    if total == 0:
        return float("nan")
    return float(mat.max(axis=0).sum() / total)


def adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mat = contingency_matrix(y_true, y_pred)
    n = mat.sum()
    if n <= 1:
        return float("nan")

    def comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) / 2.0

    nij = comb2(mat).sum()
    ai = comb2(mat.sum(axis=1)).sum()
    bj = comb2(mat.sum(axis=0)).sum()
    total = n * (n - 1) / 2.0
    expected = ai * bj / max(total, 1e-8)
    max_index = 0.5 * (ai + bj)
    denom = max(max_index - expected, 1e-8)
    return float((nij - expected) / denom)


def normalized_mutual_info(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mat = contingency_matrix(y_true, y_pred).astype(float)
    n = mat.sum()
    if n <= 0:
        return float("nan")
    pi = mat.sum(axis=1, keepdims=True)
    pj = mat.sum(axis=0, keepdims=True)
    nz = mat > 0
    mi = np.sum((mat[nz] / n) * np.log((mat[nz] * n) / (pi @ pj)[nz]))
    p_true = (pi[:, 0] / n)
    p_pred = (pj[0] / n)
    h_true = -np.sum(p_true[p_true > 0] * np.log(p_true[p_true > 0]))
    h_pred = -np.sum(p_pred[p_pred > 0] * np.log(p_pred[p_pred > 0]))
    denom = max(math.sqrt(max(h_true, 1e-8) * max(h_pred, 1e-8)), 1e-8)
    return float(mi / denom)


def best_cluster_label_match(confusion: np.ndarray) -> Dict[int, int]:
    n_true, n_pred = confusion.shape
    m = min(n_true, n_pred)
    pred_ids = list(range(n_pred))
    best_score = -1.0
    best_map: Dict[int, int] = {}

    if m <= 8:
        for perm in itertools.permutations(pred_ids, m):
            score = sum(confusion[t, perm[t]] for t in range(m))
            if score > best_score:
                best_score = float(score)
                best_map = {t: perm[t] for t in range(m)}
    else:
        remaining = set(pred_ids)
        for t in range(m):
            best_p = max(remaining, key=lambda p: confusion[t, p])
            best_map[t] = best_p
            remaining.remove(best_p)
    return best_map


def per_cluster_ess(weights: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    weighted = weights[:, None] * assignments
    numer = weighted.sum(axis=0) ** 2
    denom = (weighted ** 2).sum(axis=0) + 1e-8
    return numer / denom


def cluster_entropy(mass: np.ndarray) -> float:
    p = mass / (mass.sum() + 1e-8)
    return float(-(p * np.log(p + 1e-8)).sum())


def effective_num_clusters(mass: np.ndarray) -> float:
    p = mass / (mass.sum() + 1e-8)
    return float(np.exp(-(p * np.log(p + 1e-8)).sum()))




def plot_gaussian_ellipse(ax, mean: np.ndarray, cov: np.ndarray, color: str, n_std: float = 2.0):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-8)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    scale = n_std * np.sqrt(vals)[:, None]
    pts = (vecs @ (scale * circle)).T + mean[None, :]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.2, alpha=0.9, ls='--')

def label_centroids(points: np.ndarray, labels: np.ndarray, n_labels: int) -> np.ndarray:
    out = np.zeros((n_labels, points.shape[1]), dtype=float)
    for k in range(n_labels):
        mask = labels == k
        if np.any(mask):
            out[k] = points[mask].mean(axis=0)
        else:
            out[k] = np.nan
    return out


def plot_snapshot_panel(
    ax,
    X,
    Y,
    density,
    energy,
    batch_z: np.ndarray,
    batch_y: np.ndarray,
    particles: np.ndarray,
    logw: torch.Tensor | np.ndarray,
    centers: np.ndarray,
    tau: float,
    title: str,
    title_color=None,
    metrics: Dict[str, float] | None = None,
    true_means: np.ndarray | None = None,
    true_covs: np.ndarray | None = None,
):
    cf = ax.contourf(X, Y, density, levels=14, alpha=0.25, cmap="viridis")
    ax.contour(X, Y, energy, levels=10, linewidths=0.9, cmap="viridis")

    if particles is not None and logw is not None:
        w = normalized_weights(logw)
        a = soft_assign_points(np.asarray(particles), np.asarray(centers), tau=tau)
        cluster_id = a.argmax(axis=1)
        for c in range(centers.shape[0]):
            mask = cluster_id == c
            if np.any(mask):
                ax.scatter(
                    particles[mask, 0], particles[mask, 1],
                    s=10 + 45 * w[mask], alpha=0.20,
                    color=PALETTE[c % len(PALETTE)], edgecolors="none",
                )

    if batch_z is not None:
        for c in np.unique(batch_y):
            mask = batch_y == c
            ax.scatter(
                batch_z[mask, 0], batch_z[mask, 1],
                s=24, alpha=0.55, color=PALETTE[int(c) % len(PALETTE)],
                edgecolors="none",
            )

    if true_means is not None:
        for i in range(true_means.shape[0]):
            ax.scatter(true_means[i, 0], true_means[i, 1], marker="x", s=100, color=PALETTE[i % len(PALETTE)], linewidths=2.0, zorder=13)
            if true_covs is not None:
                plot_gaussian_ellipse(ax, true_means[i], true_covs[i], color=PALETTE[i % len(PALETTE)], n_std=2.0)

    if centers is not None:
        for i in range(centers.shape[0]):
            ax.scatter(
                centers[i, 0], centers[i, 1],
                marker="*", s=260,
                color=PALETTE[i % len(PALETTE)],
                edgecolors="black", linewidths=0.8, zorder=15,
            )

    if metrics:
        txt = f"pur={metrics.get('purity', float('nan')):.2f}  ari={metrics.get('ari', float('nan')):.2f}\ness={metrics.get('ess', float('nan')):.1f}"
        ax.text(
            0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9)
        )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title, color=title_color)
    return cf


def kde_grid_from_points(points: np.ndarray, weights: np.ndarray, X: np.ndarray, Y: np.ndarray, bandwidth: float = 0.9) -> np.ndarray:
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-8)
    diff = pts[:, None, :] - np.asarray(points)[None, :, :]
    d2 = np.sum(diff ** 2, axis=-1)
    dens = np.exp(-0.5 * d2 / max(float(bandwidth) ** 2, 1e-8)) @ w
    return dens.reshape(X.shape)


def connected_components_2d(mask: np.ndarray, min_size: int = 6):
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in nbrs:
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < H and 0 <= yy < W and mask[xx, yy] and not visited[xx, yy]:
                        visited[xx, yy] = True
                        stack.append((xx, yy))
            if len(comp) >= min_size:
                comps.append(comp)
    return comps


def threshold_for_k_components(density: np.ndarray, K: int, min_size: int = 6):
    qs = np.linspace(0.95, 0.20, 40)
    best_thr = float(np.quantile(density, 0.60))
    best_gap = 1e9
    best_comps = []
    for q in qs:
        thr = float(np.quantile(density, q))
        comps = connected_components_2d(density >= thr, min_size=min_size)
        gap = abs(len(comps) - int(K))
        if gap < best_gap:
            best_gap = gap
            best_thr = thr
            best_comps = comps
            if gap == 0:
                break
    return best_thr, best_comps


def component_centroids_from_grid(X: np.ndarray, Y: np.ndarray, density: np.ndarray, comps):
    cents = []
    masses = []
    for comp in comps:
        idx = np.array(comp, dtype=int)
        w = density[idx[:, 0], idx[:, 1]]
        xs = X[idx[:, 0], idx[:, 1]]
        ys = Y[idx[:, 0], idx[:, 1]]
        m = float(w.sum())
        masses.append(m)
        cents.append(np.array([
            float((w * xs).sum() / max(m, 1e-8)),
            float((w * ys).sum() / max(m, 1e-8)),
        ]))
    if not cents:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
    masses = np.asarray(masses, dtype=float)
    masses = masses / (masses.sum() + 1e-8)
    return np.stack(cents, axis=0), masses
