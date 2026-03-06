"""
Comprehensive Visualization for GMM SPHS / PHC Navigator.

Figures produced
----------------
  Fig 1  -- Data + true GMM vs IS-weighted particle estimates  (original space)
  Fig 2  -- Distribution evolution snapshots across streaming batches
  Fig 3  -- Mode discovery: per-component distance to truth over time
  Fig 4  -- ESS landscape: global ESS, per-cluster ESS, weight entropy
  Fig 5  -- Importance-weighted samples vs true samples  (KDE overlay)
  Fig 6  -- Navigator parameters over time (dt_scale, gamma, mass)
  Fig 7  -- Phase-space scatter of particles (PCA of phi)
  Fig 8  -- Responsibility cluster structure in embedding space (PCA of z)
  Fig 9  -- Free energy curve and convergence
  Fig 10 -- PHC topology: C(tau) curves, persistence barcodes, H_W
  Fig 11 -- Summary (paper-ready 3x3 grid)

Usage
-----
    python scripts/visualize.py \\
        --checkpoint outputs/navigator_best.pt \\
        --config     configs/gmm.yaml \\
        --K          3 \\
        --seed       42 \\
        --output     outputs/plots/
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (consistent across all figures)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [
    "#4C72B0","#DD8452","#55A868","#C44E52",
    "#8172B3","#937860","#DA8BC3","#8C8C8C",
]

plt.rcParams.update({
    "font.family":     "sans-serif",
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":      150,
    "savefig.dpi":     180,
})


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _np(t):
    if torch.is_tensor(t):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=float)


def _save(fig, path, verbose=True):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    saved -> {path}")


def _cov_ellipse(mu, cov, n_std=2.0, **kw):
    """Return a matplotlib Ellipse for a 2-D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals.clip(0))
    return Ellipse(xy=mu, width=w, height=h, angle=angle, **kw)


def _get_grid(X_np, margin=0.6, n=120):
    x0, x1 = X_np[:,0].min()-margin, X_np[:,0].max()+margin
    y0, y1 = X_np[:,1].min()-margin, X_np[:,1].max()+margin
    gx, gy = np.meshgrid(np.linspace(x0,x1,n), np.linspace(y0,y1,n))
    return gx, gy


def _gmm_density(mu_k, cov_k, pi_k, gx, gy):
    """Evaluate exact GMM density on a meshgrid."""
    pts = np.stack([gx.ravel(), gy.ravel()], -1)
    Z   = np.zeros(pts.shape[0])
    for k in range(mu_k.shape[0]):
        diff = pts - mu_k[k]
        cov  = cov_k[k] + 1e-6*np.eye(2)
        inv  = np.linalg.inv(cov)
        maha = np.einsum("nd,dd,nd->n", diff, inv, diff)
        det  = max(np.linalg.det(cov), 1e-30)
        Z   += pi_k[k] / (2*math.pi*math.sqrt(det)) * np.exp(-0.5*maha)
    return Z.reshape(gx.shape)


def _particle_gmm_density(phi, weights, K, D, gx, gy):
    """IS-average density: sum_m w_m * p(x | phi_m)."""
    from data.gmm_problem import unpack_phi, L_vec_to_matrix
    pi_t, mu, Lv = unpack_phi(phi, K, D)
    pi   = _np(torch.softmax(pi_t, -1))          # [M, K]
    L    = L_vec_to_matrix(Lv, D)
    Sig  = torch.bmm(L.reshape(-1,D,D),
                     L.reshape(-1,D,D).transpose(-1,-2)).reshape(-1,K,D,D)
    mu_np  = _np(mu)
    sig_np = _np(Sig)
    w_np   = _np(weights)
    pts    = np.stack([gx.ravel(), gy.ravel()], -1)
    G      = pts.shape[0]
    Z      = np.zeros(G)
    for m in range(mu_np.shape[0]):
        zm = np.zeros(G)
        for k in range(K):
            diff = pts - mu_np[m,k]
            cov  = sig_np[m,k] + 1e-6*np.eye(D)
            inv  = np.linalg.inv(cov)
            maha = np.einsum("nd,dd,nd->n", diff, inv, diff)
            det  = max(np.linalg.det(cov), 1e-30)
            zm  += pi[m,k] / (2*math.pi*math.sqrt(det)) * np.exp(-0.5*maha)
        Z += w_np[m] * zm
    return Z.reshape(gx.shape)


def _pca2(X_np):
    mu  = X_np.mean(0, keepdims=True)
    Xc  = X_np - mu
    cov = Xc.T @ Xc / max(len(Xc)-1, 1)
    vals, vecs = np.linalg.eigh(cov)
    idx = vals.argsort()[::-1]
    return Xc @ vecs[:, idx[:2]], vecs[:, idx[:2]], mu


def _hungarian(est, true):
    from scipy.optimize import linear_sum_assignment
    cost      = np.linalg.norm(est[:,None,:] - true[None,:,:], axis=-1)
    row, col  = linear_sum_assignment(cost)
    perm      = np.zeros(len(true), dtype=int)
    perm[row] = col
    return perm


def _weighted_mean_params(phi, weights, K, D):
    """Return IS-weighted (mu_hat [K,D], Sigma_hat [K,D,D], pi_hat [K])."""
    from data.gmm_problem import unpack_phi, L_vec_to_matrix
    pi_t, mu, Lv = unpack_phi(phi, K, D)
    pi   = torch.softmax(pi_t, -1)                           # [M, K]
    L    = L_vec_to_matrix(Lv, D)
    Sig  = torch.bmm(L.reshape(-1,D,D),
                     L.reshape(-1,D,D).transpose(-1,-2)).reshape(-1,K,D,D)
    w    = weights
    mu_hat  = (w[:,None,None]*mu).sum(0)                     # [K, D]
    sig_hat = (w[:,None,None,None]*Sig).sum(0)               # [K, D, D]
    pi_hat  = (w[:,None]*pi).sum(0)                          # [K]
    return _np(mu_hat), _np(sig_hat), _np(pi_hat)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 1 — Data + true GMM vs IS-weighted estimates
# ═════════════════════════════════════════════════════════════════════════════

def fig1_data_vs_estimate(problem, phi_final, weights_final, K, D, path):
    X_np       = _np(problem.X)
    mu_true    = _np(problem.mu_true)
    sig_true   = _np(problem.Sigma_true)
    pi_true    = _np(problem.pi_true)
    lab        = _np(problem.labels_true).astype(int)

    gx, gy  = _get_grid(X_np)
    Z_true  = _gmm_density(mu_true, sig_true, pi_true, gx, gy)
    mu_hat, sig_hat, pi_hat = _weighted_mean_params(phi_final, weights_final, K, D)
    Z_est   = _particle_gmm_density(phi_final, weights_final, K, D, gx, gy)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # -- Left: true -------------------------------------------------------------
    ax = axes[0]
    for k in range(K):
        mask = lab == k
        ax.scatter(X_np[mask,0], X_np[mask,1], s=6, alpha=0.35,
                   color=PALETTE[k%len(PALETTE)], label=f"C{k}")
        try:
            el = _cov_ellipse(mu_true[k], sig_true[k],
                              facecolor="none",
                              edgecolor=PALETTE[k%len(PALETTE)],
                              linewidth=2, linestyle="--")
            ax.add_patch(el)
        except Exception:
            pass
        ax.scatter(*mu_true[k], marker="*", s=260,
                   color=PALETTE[k%len(PALETTE)], zorder=10, edgecolors="k")
    ax.contour(gx, gy, Z_true, levels=5, colors="k", linewidths=0.6, alpha=0.5)
    ax.set_title("True GMM  p*(x)"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(markerscale=2, ncol=2)

    # -- Middle: IS estimate ----------------------------------------------------
    ax = axes[1]
    ax.scatter(X_np[:,0], X_np[:,1], s=4, alpha=0.15, color="#bbbbbb")
    cf = ax.contourf(gx, gy, Z_est, levels=18, cmap="Blues", alpha=0.55)
    ax.contour(gx, gy, Z_est, levels=5, colors="#08306b", linewidths=0.8)
    fig.colorbar(cf, ax=ax, label="p_hat(x)", shrink=0.85)
    for k in range(K):
        try:
            el = _cov_ellipse(mu_hat[k], sig_hat[k],
                              facecolor="none",
                              edgecolor=PALETTE[k%len(PALETTE)],
                              linewidth=2.0)
            ax.add_patch(el)
        except Exception:
            pass
        ax.scatter(*mu_hat[k],  marker="^", s=180,
                   color=PALETTE[k%len(PALETTE)], zorder=10, edgecolors="k",
                   label=f"mu_hat_{k}")
        ax.scatter(*mu_true[k], marker="*", s=160,
                   color=PALETTE[k%len(PALETTE)], zorder=10,
                   edgecolors="red", alpha=0.8)
    ax.set_title("IS-Weighted Estimate  (^ = estimated,  * = true)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(markerscale=1.5, ncol=2)

    # -- Right: density difference ----------------------------------------------
    ax = axes[2]
    diff = Z_est - Z_true
    vmax = max(np.abs(diff).max(), 1e-9)
    cf2 = ax.contourf(gx, gy, diff, levels=20,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.contour(gx, gy, Z_true, levels=4, colors="k", linewidths=0.5, alpha=0.4)
    fig.colorbar(cf2, ax=ax, label="p_hat - p*", shrink=0.85)
    ax.set_title("Density Error  p_hat - p*\n(blue=under, red=over)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    fig.suptitle(f"GMM SPHS -- Data vs Estimate  (K={K},  N={len(X_np)})",
                 fontweight="bold")
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 2 — Distribution evolution snapshots
# ═════════════════════════════════════════════════════════════════════════════

def fig2_distribution_evolution(problem, phi_history, weights_history,
                                  batches_out, K, D, path, n_snap=6):
    X_np      = _np(problem.X)
    mu_true   = _np(problem.mu_true)
    T_traj    = len(phi_history)
    T_batches = len(batches_out)
    snaps     = np.linspace(0, T_traj-1, n_snap, dtype=int)
    gx, gy    = _get_grid(X_np, margin=0.8)

    cols = 3
    rows = math.ceil(n_snap / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 4.5*rows))
    axes_flat = axes.flatten() if n_snap > 1 else [axes]

    cmap_t = cm.plasma
    norm_t = Normalize(vmin=0, vmax=T_traj-1)

    for pi, t in enumerate(snaps):
        ax  = axes_flat[pi]
        phi = phi_history[t]
        w   = weights_history[t]

        # grey background
        ax.scatter(X_np[:,0], X_np[:,1], s=4, alpha=0.12, color="#cccccc", zorder=0)

        # current batch (if t > 0)
        bi = min(t-1, T_batches-1)
        if t > 0 and bi >= 0:
            Xb = _np(batches_out[bi])
            ax.scatter(Xb[:,0], Xb[:,1], s=16, alpha=0.6,
                       color=cmap_t(norm_t(t)), zorder=1)

        # IS density
        Z = _particle_gmm_density(phi, w, K, D, gx, gy)
        ax.contourf(gx, gy, Z, levels=12, cmap="Blues", alpha=0.45, zorder=2)
        ax.contour(gx, gy, Z, levels=5, colors=["#0a2f6e"], linewidths=0.7, zorder=3)

        # true means
        for k in range(K):
            ax.scatter(*mu_true[k], marker="*", s=220,
                       color=PALETTE[k%len(PALETTE)], zorder=10, edgecolors="k",
                       linewidths=0.5)

        lbl = "Init" if t == 0 else f"After batch {t}"
        ax.set_title(f"t = {t}  ({lbl})", color=cmap_t(norm_t(t)))
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.tick_params(labelsize=7)

    for j in range(pi+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    sm = cm.ScalarMappable(cmap=cmap_t, norm=norm_t)
    sm.set_array([])
    fig.colorbar(sm, ax=axes_flat, label="Trajectory step t",
                 fraction=0.015, pad=0.04)
    fig.suptitle(f"Distribution Evolution over Streaming Batches  (K={K})",
                 fontweight="bold", y=1.01)
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 3 — Mode discovery dynamics
# ═════════════════════════════════════════════════════════════════════════════

def fig3_mode_discovery(problem, phi_history, weights_history, K, D, path):
    from data.gmm_problem import unpack_phi
    mu_true_np = _np(problem.mu_true)
    pi_true_np = _np(problem.pi_true)
    T = len(phi_history)

    dist_mat = np.zeros((T, K))
    pi_mat   = np.zeros((T, K))

    for t, (phi, w) in enumerate(zip(phi_history, weights_history)):
        pi_t, mu, _ = unpack_phi(phi, K, D)
        pi_est = torch.softmax(pi_t, -1)
        mu_w   = _np((w[:,None,None]*mu).sum(0))
        pi_w   = _np((w[:,None]*pi_est).sum(0))
        try:
            perm         = _hungarian(mu_w, mu_true_np)
            dist_mat[t]  = np.linalg.norm(mu_w - mu_true_np[perm], axis=-1)
            pi_mat[t]    = pi_w[perm]
        except Exception:
            dist_mat[t] = np.nan
            pi_mat[t]   = pi_w

    ts  = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    for k in range(K):
        ax.plot(ts, dist_mat[:,k], "-o", ms=4,
                color=PALETTE[k%len(PALETTE)], lw=1.8, label=f"Comp {k}")
    ax.axhline(0,   color="k",     ls="--", lw=0.8, alpha=0.35)
    ax.fill_between(ts, 0, 0.2, alpha=0.08, color="green",
                    label="Tolerance 0.2")
    ax.set_ylabel("||mu_hat_k - mu*_k||")
    ax.set_title("Per-Component Mean Estimation Error")
    ax.legend(ncol=K); ax.set_ylim(bottom=0)

    ax = axes[1]
    for k in range(K):
        ax.plot(ts, pi_mat[:,k], "-o", ms=4,
                color=PALETTE[k%len(PALETTE)], lw=1.8, label=f"pi_hat_{k}")
        ax.axhline(pi_true_np[k], color=PALETTE[k%len(PALETTE)],
                   ls=":", lw=1.5, alpha=0.6)
    ax.set_xlabel("Trajectory step  t")
    ax.set_ylabel("Mixing weight  pi_hat_k")
    ax.set_title("Mixing Weight Evolution  (dotted = true pi*_k)")
    ax.set_ylim(0, 1); ax.legend(ncol=K)

    fig.suptitle("Mode Discovery Dynamics", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 4 — ESS landscape
# ═════════════════════════════════════════════════════════════════════════════

def fig4_ess_landscape(topo_history, ess_curve, weights_history, K, path):
    T     = len(topo_history)
    C_max = topo_history[0].C_max if topo_history else 8
    M     = len(weights_history[0]) if weights_history else 1
    ts    = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=False)

    # row 0: global ESS
    ax = axes[0]
    ev = np.array(ess_curve)
    ax.fill_between(ts, 0, ev, alpha=0.22, color="#4C72B0")
    ax.plot(ts, ev, "-o", ms=4, color="#4C72B0", lw=2, label="Global ESS")
    ax.axhline(M,     color="green",  ls="--", lw=1.4,
               label=f"Max ESS = M = {M}")
    ax.axhline(0.5*M, color="orange", ls="--", lw=1.4,
               label=f"Resample thresh = 0.5M = {0.5*M:.0f}")
    ax.set_ylabel("ESS"); ax.set_title("Global Effective Sample Size")
    ax.legend(); ax.set_ylim(bottom=0)

    # row 1: per-cluster ESS
    ax = axes[1]
    for c in range(min(K, C_max)):
        ec = [float(_np(d.ESS_c[c])) for d in topo_history]
        ax.plot(ts, ec, "-o", ms=3,
                color=PALETTE[c%len(PALETTE)], lw=1.8, label=f"Cluster {c}")
    ax.axhline(5.0, color="red", ls=":", lw=1.5, label="ESS_min = 5")
    ax.set_ylabel("Per-cluster ESS")
    ax.set_title("Cluster Sampling Adequacy  (PHC responsibility clusters)")
    ax.legend(ncol=K); ax.set_ylim(bottom=0)

    # row 2: weight distribution violin
    ax = axes[2]
    snap_idx = np.linspace(0, len(weights_history)-1,
                            min(8, len(weights_history)), dtype=int)
    data_v = [_np(weights_history[t]) for t in snap_idx]
    vp = ax.violinplot(data_v, positions=snap_idx.tolist(), widths=0.8,
                        showmedians=True, showextrema=True)
    for body in vp["bodies"]:
        body.set_alpha(0.45); body.set_facecolor("#4C72B0")
    unif = 1.0 / M
    ax.axhline(unif, color="green", ls="--", lw=1.2,
               label=f"Uniform 1/M = {unif:.3f}")
    ax.set_xlabel("Trajectory step  t")
    ax.set_ylabel("Particle weight w_m")
    ax.set_title("Weight Distribution  (narrow + centred on 1/M = good mixing)")
    ax.legend()

    # row 3: cluster mass stacked area
    ax = axes[3]
    W = np.array([[float(_np(d.W_c[c])) for c in range(min(K, C_max))]
                   for d in topo_history])   # [T, K]
    ax.stackplot(ts, W.T,
                 labels=[f"Cluster {c}" for c in range(W.shape[1])],
                 colors=PALETTE[:W.shape[1]], alpha=0.8)
    ax.set_xlabel("Streaming batch"); ax.set_ylabel("Cluster mass W_c")
    ax.set_title("Cluster Mass Balance  (equal-width bands = balanced)")
    ax.legend(ncol=K, loc="upper right"); ax.set_ylim(0, 1)

    fig.suptitle("ESS and Sampling Quality Diagnostics", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 5 — IS samples vs true samples
# ═════════════════════════════════════════════════════════════════════════════

def fig5_is_vs_true(problem, phi_final, weights_final, K, D, n_samples, path):
    from data.gmm_problem import unpack_phi, L_vec_to_matrix

    X_true_np = _np(problem.X)[:n_samples]
    mu_true   = _np(problem.mu_true)
    sig_true  = _np(problem.Sigma_true)
    pi_true   = _np(problem.pi_true)

    # Draw IS samples from particle ensemble
    w_np  = _np(weights_final).clip(0)
    w_np /= w_np.sum() + 1e-30
    idx   = np.random.choice(len(w_np), size=min(n_samples, len(w_np)),
                              replace=True, p=w_np)

    pi_t, mu, Lv = unpack_phi(phi_final, K, D)
    pi_est = _np(torch.softmax(pi_t, -1))   # [M, K]
    L      = L_vec_to_matrix(Lv, D)
    Sig    = _np(torch.bmm(L.reshape(-1,D,D),
                           L.reshape(-1,D,D).transpose(-1,-2)).reshape(-1,K,D,D))
    mu_np  = _np(mu)

    is_pts = []
    for i in idx:
        pk = pi_est[i] / pi_est[i].sum()
        k  = np.random.choice(K, p=pk)
        s  = np.random.multivariate_normal(mu_np[i,k],
                                            Sig[i,k] + 1e-5*np.eye(D))
        is_pts.append(s)
    is_pts = np.array(is_pts)

    gx, gy   = _get_grid(X_true_np, margin=0.8)
    Z_true   = _gmm_density(mu_true, sig_true, pi_true, gx, gy)
    mu_hat, sig_hat, pi_hat = _weighted_mean_params(phi_final, weights_final, K, D)
    # Particle mixture: sum_m w_m p(x | phi_m)  (NOT p(x | mean_m w_m phi_m))
    # Average-of-parameters != average-of-distributions for GMMs.
    Z_est    = _particle_gmm_density(phi_final, weights_final, K, D, gx, gy)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.scatter(X_true_np[:,0], X_true_np[:,1], s=6, alpha=0.4, color="#4C72B0")
    ax.contour(gx, gy, Z_true, levels=6, colors="k", linewidths=1.0)
    for k in range(K):
        ax.scatter(*mu_true[k], marker="*", s=260,
                   color=PALETTE[k%len(PALETTE)], zorder=10, edgecolors="k")
    ax.set_title(f"True samples  (N={len(X_true_np)})")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    ax = axes[1]
    ax.scatter(is_pts[:,0], is_pts[:,1], s=6, alpha=0.4, color="#DD8452")
    ax.contour(gx, gy, Z_est, levels=6, colors="k", linewidths=1.0)
    for k in range(K):
        ax.scatter(*mu_hat[k], marker="^", s=200,
                   color=PALETTE[k%len(PALETTE)], zorder=10, edgecolors="k")
    ax.set_title(f"IS-weighted samples  (N={len(is_pts)})")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    ax = axes[2]
    diff = Z_est - Z_true
    vmax = max(np.abs(diff).max(), 1e-9)
    cf   = ax.contourf(gx, gy, diff, levels=20, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
    ax.contour(gx, gy, Z_true, levels=4, colors="k", linewidths=0.5, alpha=0.45)
    fig.colorbar(cf, ax=ax, label="p_hat(x) - p*(x)", shrink=0.85)
    ax.set_title("Density Difference  p_hat - p*")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    fig.suptitle("IS-Weighted Samples vs True Samples", fontweight="bold")
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 6 — Navigator parameters over time
# ═════════════════════════════════════════════════════════════════════════════

def fig6_navigator_params(params_history, path):
    if not params_history:
        return
    ts       = [p["batch"] for p in params_history]
    # v14+: beta renamed to dt_scale (target-neutral timestep multiplier)
    dt_scales = [p.get("dt_scale", p.get("beta", 1.0)) for p in params_history]
    gams     = [p["gamma"] for p in params_history]
    md       = np.array([p["M_diag_mean"] for p in params_history])
    ms       = np.array([p["M_diag_std"]  for p in params_history])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(ts, dt_scales, "-o", ms=4, color="#4C72B0", lw=2)
    lo = min(dt_scales) * 0.9 if min(dt_scales) > 0 else 0.0
    ax.fill_between(ts, lo, dt_scales, alpha=0.18, color="#4C72B0")
    ax.axhline(np.mean(dt_scales), color="red", ls="--", lw=1.2,
               label=f"Mean dt_scale = {np.mean(dt_scales):.2f}")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.6, label="dt_scale = 1 (identity)")
    ax.set_ylabel("dt_scale  (timestep multiplier)")
    ax.set_title("Timestep Multiplier  (controls mixing speed, not target distribution)")
    ax.legend()

    ax = axes[1]
    ax.plot(ts, gams, "-o", ms=4, color="#55A868", lw=2)
    ax.fill_between(ts, min(gams)*0.9, gams, alpha=0.18, color="#55A868")
    ax.axhline(1.0, color="red", ls="--", lw=1.2, label="gamma = 1.0 (identity)")
    ax.set_ylabel("gamma  (friction)")
    ax.set_title("Langevin Friction  (high = overdamped / diffusive)")
    ax.legend()

    ax = axes[2]
    ax.plot(ts, md, "-o", ms=4, color="#C44E52", lw=2, label="Mean M_diag")
    ax.fill_between(ts, md-ms, md+ms, alpha=0.18, color="#C44E52",
                    label="+/-1 std across dims")
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.45, label="Identity mass")
    ax.set_xlabel("Streaming batch  t")
    ax.set_ylabel("M_diag (mean)")
    ax.set_title("Diagonal Mass Matrix  (preconditioning BAOAB sampler)")
    ax.legend()

    fig.suptitle("Transformer Navigator Outputs over Streaming Batches",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 7 — Phase-space scatter (PCA of phi)
# ═════════════════════════════════════════════════════════════════════════════

def fig7_phase_space(phi_history, weights_history, K, D, path, n_snap=6):
    from data.gmm_problem import unpack_phi
    T     = len(phi_history)
    snaps = np.linspace(0, T-1, min(n_snap, T), dtype=int)

    # PCA fitted on mu subspace of all snapshots
    all_mu = np.concatenate([
        _np(phi_history[t])[:, K:K+K*D] for t in snaps
    ], axis=0)
    try:
        _, evecs, mu_c = _pca2(all_mu)
    except Exception:
        return

    cols = 3
    rows = math.ceil(len(snaps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 4.5*rows))
    axes_flat = axes.flatten() if len(snaps) > 1 else [axes]

    cmap_t = cm.plasma
    norm_t = Normalize(vmin=0, vmax=T-1)

    for pi, t in enumerate(snaps):
        ax  = axes_flat[pi]
        phi = _np(phi_history[t])
        w   = _np(weights_history[t])
        M   = phi.shape[0]

        mu_sub = phi[:, K:K+K*D] - mu_c
        xy     = mu_sub @ evecs[:, :2]

        # colour by dominant component (argmax column norm across D dims)
        mu_r   = phi[:, K:K+K*D].reshape(M, K, D)
        comp   = np.linalg.norm(mu_r, axis=-1).argmax(axis=1)
        colors = [PALETTE[c%len(PALETTE)] for c in comp]

        w_norm = w / (w.max() + 1e-30)
        sizes  = 15 + 90 * w_norm

        ax.scatter(xy[:,0], xy[:,1], s=sizes, c=colors, alpha=0.65,
                   edgecolors="none")
        lbl = "Init" if t == 0 else f"Batch {t}"
        ax.set_title(f"t = {t}  ({lbl})", color=cmap_t(norm_t(t)))
        ax.set_xlabel("PC1 (mu-subspace)"); ax.set_ylabel("PC2")
        ax.tick_params(labelsize=7)

    for j in range(pi+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Phase-Space Scatter of Particles  (PCA of mu-subspace of phi)\n"
        "Size proportional to importance weight,  colour = dominant component",
        fontweight="bold", y=1.01)
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 8 — Responsibility embedding cluster structure
# ═════════════════════════════════════════════════════════════════════════════

def fig8_embedding_clusters(phi_history, weights_history, batches_out,
                              K, D, path, n_snap=4):
    from models.gmm_embedder import GMMEmbedder
    from data.gmm_problem import unpack_phi

    T     = len(phi_history)
    snaps = np.linspace(1, T-1, min(n_snap, T-1), dtype=int)

    cols = min(n_snap, 2)
    rows = math.ceil(len(snaps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.5*cols, 5*rows))
    axes_flat = axes.flatten() if len(snaps) > 1 else [axes]

    cmap_t = cm.plasma
    norm_t = Normalize(vmin=0, vmax=T-1)

    for pi, t in enumerate(snaps):
        ax    = axes_flat[pi]
        phi   = phi_history[t]
        w     = weights_history[t]
        bi    = min(t-1, len(batches_out)-1)
        X_ref = batches_out[bi] if bi >= 0 else None

        try:
            embedder = GMMEmbedder(K=K, D=D)
            with torch.no_grad():
                z = embedder.embed(phi, X_ref=X_ref)
            z_np = _np(z)
            try:
                xy, _, _ = _pca2(z_np)
            except Exception:
                xy = z_np[:, :2]

            w_np   = _np(w)
            w_norm = w_np / (w_np.max() + 1e-30)
            sizes  = 16 + 75 * w_norm

            # colour by responsibility argmax (last K dims when X_ref used)
            if X_ref is not None and z_np.shape[-1] >= K:
                resp  = z_np[:, -K:]
                comp  = resp.argmax(axis=1)
            else:
                pi_t, _, _ = unpack_phi(phi, K, D)
                comp = _np(torch.softmax(pi_t, -1)).argmax(axis=1)

            colors = [PALETTE[c%len(PALETTE)] for c in comp]
            ax.scatter(xy[:,0], xy[:,1], s=sizes, c=colors,
                       alpha=0.7, edgecolors="none")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)

        ax.set_title(f"Embedding  t = {t}", color=cmap_t(norm_t(t)))
        ax.set_xlabel("PC1  (embedding)"); ax.set_ylabel("PC2  (embedding)")
        ax.tick_params(labelsize=7)

    for j in range(pi+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Responsibility Embedding Cluster Structure  (PCA)\n"
        "Size proportional to importance weight,  colour = dominant component",
        fontweight="bold")
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 9 — Free energy and convergence
# ═════════════════════════════════════════════════════════════════════════════

def fig9_energy_convergence(energy_curve, ess_curve, topo_history, K, path):
    T    = len(energy_curve)
    ts   = np.arange(T)
    H_W  = [d.H_W for d in topo_history]
    pers = [d.max_persistence for d in topo_history]
    bar  = [d.barrier_mean for d in topo_history]
    H_target = math.log(K) if K > 1 else 0.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]
    ax.plot(ts, energy_curve, "-o", ms=4, color="#4C72B0", lw=2,
            label="Free energy  U(phi_bar; X_t)")
    ax.fill_between(ts, min(energy_curve)*1.05, energy_curve,
                    alpha=0.14, color="#4C72B0")
    ax.plot(ts, np.minimum.accumulate(energy_curve), "--",
            color="green", lw=1.5, label="Running minimum")
    ax.set_ylabel("Free Energy  U"); ax.legend()
    ax.set_title("Weighted-Mean Free Energy over Streaming Batches")

    ax = axes[1]
    ax.plot(ts, H_W, "-o", ms=4, color="#55A868", lw=2,
            label="H_W  (cluster mass entropy)")
    ax.axhline(H_target, color="green", ls="--", lw=1.5,
               label=f"Ideal  log({K}) = {H_target:.2f}")
    ax2 = ax.twinx()
    ax2.plot(ts, pers, "-s", ms=4, color="#C44E52", lw=1.5,
             alpha=0.75, label="Max persistence")
    ax2.set_ylabel("Max persistence", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax.set_ylabel("H_W  (nats)"); ax.legend(loc="lower left")
    ax2.legend(loc="lower right")
    ax.set_title("Cluster Balance (H_W) and Topological Complexity (persistence)")

    ax = axes[2]
    ax.plot(ts, bar, "-o", ms=4, color="#8172B3", lw=2,
            label="Mean barrier height")
    ax.fill_between(ts, 0, bar, alpha=0.18, color="#8172B3")
    ax.set_ylabel("Barrier height  (H^clust units)")
    ax.set_xlabel("Streaming batch  t")
    ax.set_title("Mean Intercluster Barrier  (higher = more separated modes)")
    ax.legend()

    fig.suptitle("Free Energy Convergence and Topological Diagnostics",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 10 — PHC topology diagnostics
# ═════════════════════════════════════════════════════════════════════════════

def fig10_phc_topology(filtration_history, topo_history, K, path):
    T = len(filtration_history)

    fig = plt.figure(figsize=(17, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # -- C(tau) curves (full left column) ----------------------------------------
    ax_c = fig.add_subplot(gs[:, 0])
    cmap = cm.plasma
    norm = Normalize(vmin=0, vmax=T-1)
    for t, fr in enumerate(filtration_history):
        tau_np = _np(fr.tau_grid)
        C_np   = _np(fr.C_tau_curve)
        ax_c.plot(tau_np, C_np,
                  color=cmap(norm(t)),
                  alpha=0.4 + 0.6*t/max(T-1,1),
                  lw=1.4)
    ax_c.axhline(K, color="red", ls="--", lw=2.0, label=f"K* = {K}")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax_c, label="Batch t", shrink=0.75)
    ax_c.set_xlabel("tau  (H^clust threshold)")
    ax_c.set_ylabel("C(tau)  (# components)")
    ax_c.set_title("C(tau) Curves\n(bright = late batches)")
    ax_c.legend()

    # -- Persistence barcodes at 4 snapshots (right 2x2) -------------------------
    snaps = np.linspace(0, T-1, 4, dtype=int)
    sub   = [fig.add_subplot(gs[i//2, 1 + i%2]) for i in range(4)]

    for ai, t in enumerate(snaps):
        ax   = sub[ai]
        fr   = filtration_history[t]
        bcd  = fr.barcode
        Hmax = float(_np(fr.H_vals).max()) * 1.1 if fr.H_vals.numel() else 1.0

        for ci, (birth, death) in enumerate(bcd):
            d     = min(death, Hmax) if not math.isinf(death) else Hmax
            color = "#4C72B0" if math.isinf(death) else "#aaaaaa"
            ax.barh(ci, d - birth, left=birth, height=0.55,
                    color=color, alpha=0.8)

        if len(fr.tau_grid) > 0:
            tau_op = float(_np(fr.tau_grid)[len(fr.tau_grid)//2])
            ax.axvline(tau_op, color="red", ls=":", lw=1.5)

        ax.set_title(f"Barcodes  t = {t}")
        ax.set_xlabel("tau"); ax.set_ylabel("Component")

    fig.suptitle(f"PHC Topological Diagnostics  (K={K})", fontweight="bold")
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 11 — Summary (paper-ready 3x3)
# ═════════════════════════════════════════════════════════════════════════════

def fig11_summary(problem, phi_history, weights_history, topo_history,
                   batches_out, K, D, path):
    from data.gmm_problem import unpack_phi, L_vec_to_matrix

    X_np    = _np(problem.X)
    mu_true = _np(problem.mu_true)
    sig_true= _np(problem.Sigma_true)
    pi_true = _np(problem.pi_true)
    T       = len(phi_history)

    gx, gy  = _get_grid(X_np, margin=0.7, n=100)
    Z_true  = _gmm_density(mu_true, sig_true, pi_true, gx, gy)
    phi_f   = phi_history[-1]
    w_f     = weights_history[-1]
    Z_est   = _particle_gmm_density(phi_f, w_f, K, D, gx, gy)
    mu_hat, sig_hat, pi_hat = _weighted_mean_params(phi_f, w_f, K, D)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # [0,0] True GMM
    ax = axes[0, 0]
    ax.scatter(X_np[:,0], X_np[:,1], s=4, alpha=0.18, color="#bbbbbb")
    ax.contourf(gx, gy, Z_true, levels=10, cmap="Blues", alpha=0.55)
    ax.contour(gx, gy, Z_true, levels=5, colors="k", linewidths=0.7)
    for k in range(K):
        ax.scatter(*mu_true[k], marker="*", s=220,
                   color=PALETTE[k%len(PALETTE)], zorder=10, edgecolors="k")
    ax.set_title("True GMM  p*(x)"); ax.set_xlabel("x1"); ax.set_ylabel("x2")

    # [0,1] IS estimate
    ax = axes[0, 1]
    ax.scatter(X_np[:,0], X_np[:,1], s=4, alpha=0.12, color="#bbbbbb")
    ax.contourf(gx, gy, Z_est, levels=10, cmap="Oranges", alpha=0.55)
    ax.contour(gx, gy, Z_est, levels=5, colors="k", linewidths=0.7)
    for k in range(K):
        ax.scatter(*mu_hat[k],  marker="^", s=180,
                   color=PALETTE[k%len(PALETTE)], zorder=10, edgecolors="k")
        ax.scatter(*mu_true[k], marker="*", s=150,
                   color=PALETTE[k%len(PALETTE)], zorder=10,
                   edgecolors="red", alpha=0.7)
    ax.set_title("IS Estimate  (^ estimated,  * true)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    # [0,2] Density error
    ax = axes[0, 2]
    diff = Z_est - Z_true
    vmax = max(np.abs(diff).max(), 1e-9)
    ax.contourf(gx, gy, diff, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.contour(gx, gy, Z_true, levels=4, colors="k", linewidths=0.5, alpha=0.4)
    ax.set_title("Density Error  p_hat - p*"); ax.set_xlabel("x1"); ax.set_ylabel("x2")

    # [1,0] Phase-space PCA
    ax = axes[1, 0]
    cmap_t = cm.plasma
    step   = max(1, T//6)
    all_mu = np.concatenate([_np(phi_history[t])[:, K:K+K*D]
                              for t in range(0, T, step)], axis=0)
    try:
        _, evecs, mu_c = _pca2(all_mu)
        cumM = 0
        for ti, t in enumerate(range(0, T, step)):
            phi_t  = _np(phi_history[t])
            mu_sub = phi_t[:, K:K+K*D] - mu_c
            xy     = mu_sub @ evecs[:, :2]
            ax.scatter(xy[:,0], xy[:,1], s=8, alpha=0.5,
                       color=cmap_t(ti / max(T//step, 1)),
                       edgecolors="none")
        ax.set_title("Phase-Space PCA  (bright = late)")
    except Exception:
        ax.text(0.5, 0.5, "PCA unavailable", transform=ax.transAxes, ha="center")
    ax.set_xlabel("PC1  (phi_mu)"); ax.set_ylabel("PC2")

    # [1,1] H_W convergence
    ax = axes[1, 1]
    H_W = [d.H_W for d in topo_history]
    ax.plot(range(len(H_W)), H_W, "-o", ms=4, color="#55A868", lw=2)
    H_target = math.log(K) if K > 1 else 0.0
    ax.axhline(H_target, color="red", ls="--", lw=1.5,
               label=f"Ideal log({K}) = {H_target:.2f}")
    ax.set_title("Cluster Mass Entropy  H_W"); ax.set_xlabel("Batch")
    ax.set_ylabel("H_W  (nats)"); ax.legend()

    # [1,2] Global ESS
    ax = axes[1, 2]
    g_ess = [d.global_ESS for d in topo_history]
    M_val = len(weights_history[0])
    ax.fill_between(range(len(g_ess)), 0, g_ess, alpha=0.25, color="#4C72B0")
    ax.plot(g_ess, "-o", ms=4, color="#4C72B0", lw=2, label="Global ESS")
    ax.axhline(0.5*M_val, color="orange", ls="--", label=f"0.5M = {0.5*M_val:.0f}")
    ax.axhline(M_val,     color="green",  ls="--", label=f"M = {M_val}")
    ax.set_title("Global ESS"); ax.set_xlabel("Batch"); ax.set_ylabel("ESS")
    ax.legend(fontsize=7); ax.set_ylim(bottom=0)

    # [2,0] C(tau) at quantile grid (4 snapshots)
    ax = axes[2, 0]
    snaps_c = np.linspace(0, len(topo_history)-1, 4, dtype=int)
    cols_c  = ["#cce5ff","#6baed6","#2171b5","#08306b"]
    for ci2, t in enumerate(snaps_c):
        d   = topo_history[t]
        qs  = _np(d.C_at_quantiles)
        qxs = np.linspace(0, 1, len(qs))
        ax.plot(qxs, qs, "-o", ms=4, color=cols_c[ci2], lw=2, label=f"t={t}")
    ax.axhline(K, color="red", ls="--", lw=1.5, label=f"K*={K}")
    ax.set_xlabel("tau quantile"); ax.set_ylabel("C(tau)")
    ax.set_title("C(tau) at quantile grid"); ax.legend(fontsize=7)

    # [2,1] Mean mode error
    ax = axes[2, 1]
    dists = []
    for t in range(T):
        phi = phi_history[t]; w = weights_history[t]
        from data.gmm_problem import unpack_phi as _up
        _, mu_t, _ = _up(phi, K, D)
        mu_w = _np((w[:,None,None]*mu_t).sum(0))
        try:
            perm = _hungarian(mu_w, mu_true)
            d    = np.linalg.norm(mu_w - mu_true[perm], axis=-1).mean()
        except Exception:
            d = np.nan
        dists.append(d)
    ax.plot(range(T), dists, "-o", ms=4, color="#C44E52", lw=2)
    ax.fill_between(range(T), 0, dists, alpha=0.18, color="#C44E52")
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.35)
    ax.set_title("Mean Mode Error  ||mu_hat - mu*||")
    ax.set_xlabel("Trajectory step  t"); ax.set_ylabel("Distance")
    ax.set_ylim(bottom=0)

    # [2,2] Min cluster ESS + barrier
    ax = axes[2, 2]
    min_ess = [d.min_ESS_c for d in topo_history]
    bar_v   = [d.barrier_mean for d in topo_history]
    ax.plot(range(len(min_ess)), min_ess, "-o", ms=4, color="#8172B3",
            lw=2, label="min ESS_c")
    ax.axhline(5.0, color="red", ls=":", lw=1.5, label="ESS_min = 5")
    ax2 = ax.twinx()
    ax2.plot(range(len(bar_v)), bar_v, "--s", ms=3, color="#937860",
             lw=1.5, alpha=0.8, label="Mean barrier")
    ax2.set_ylabel("Barrier height", color="#937860")
    ax2.tick_params(axis="y", labelcolor="#937860")
    ax.set_title("Min Cluster ESS & Barrier Height")
    ax.set_xlabel("Batch"); ax.set_ylabel("min ESS_c", color="#8172B3")
    ax.legend(fontsize=7); ax2.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        f"GMM SPHS Navigator Summary -- K={K},  D={D}",
        fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",  default="configs/gmm.yaml")
    parser.add_argument("--K",       type=int, default=3)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  default="outputs/plots/")
    parser.add_argument("--device",  default=None)
    parser.add_argument("--n_snap",  type=int, default=6,
                        help="Number of snapshots in evolution/phase-space figures")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device(args.device or cfg.get("device", "cpu"))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load navigator ----------------------------------------------------------
    from scripts.train import build_navigator
    from data.gmm_problem import phi_dim as compute_phi_dim

    D    = cfg["problem"]["D"]
    ckpt = torch.load(args.checkpoint, map_location=device)

    ckpt_cfg   = ckpt.get("config", {})
    feat_dim   = ckpt_cfg.get("feat_dim",  None)
    phi_dim_ck = ckpt_cfg.get("phi_dim",   None)

    if feat_dim is None or phi_dim_ck is None:
        from scripts.train import compute_feat_dim
        feat_dim   = compute_feat_dim(cfg)
        phi_dim_ck = compute_phi_dim(cfg["problem"]["K_max"], D)
        print(f"[warn] recomputing dims: feat_dim={feat_dim}, phi_dim={phi_dim_ck}")
    else:
        print(f"Checkpoint dims: feat_dim={feat_dim}, phi_dim={phi_dim_ck}")

    # Patch cfg with checkpoint-stored navigator params so build_navigator
    # reconstructs the exact same architecture that was trained.
    # Falls back to YAML values for fields not yet in older checkpoints.
    if "C_max" in ckpt_cfg:
        cfg["navigator"]["C_max"] = ckpt_cfg["C_max"]
    if "control_mode" in ckpt_cfg:
        cfg["control"]["mode"] = ckpt_cfg["control_mode"]
    if "dt_scale_min" in ckpt_cfg:
        cfg["navigator"]["dt_scale_min"] = ckpt_cfg["dt_scale_min"]
    if "dt_scale_max" in ckpt_cfg:
        cfg["navigator"]["dt_scale_max"] = ckpt_cfg["dt_scale_max"]

    navigator = build_navigator(cfg, feat_dim, phi_dim_ck).to(device)
    navigator.load_state_dict(ckpt["navigator_state"])
    navigator.eval()

    chol_sz   = D * (D + 1) // 2
    K_nav_max = phi_dim_ck // (1 + D + chol_sz)
    K = args.K
    if K > K_nav_max:
        print(f"[warn] --K {K} > K_nav_max={K_nav_max}; clamping.")
        K = K_nav_max

    # -- Sample problem ----------------------------------------------------------
    from data.gmm_problem import (sample_gmm_problem, get_streaming_batches,
                                   sample_prior_particles)

    problem  = sample_gmm_problem(K=K, D=D, N=cfg["problem"]["N"],
                                   overlap=0.3, device=device)
    batches  = get_streaming_batches(problem,
                                      cfg["problem"]["T_batches"],
                                      cfg["problem"]["batch_size"])
    phi_init = sample_prior_particles(cfg["particle"]["n_particles"],
                                       K, D, problem, device=device)

    # -- Run episode with full trajectory ----------------------------------------
    from training.episode_trainer import EpisodeTrainer

    problem_cfg = cfg["problem"].copy()
    problem_cfg["n_particles"] = cfg["particle"]["n_particles"]
    trainer = EpisodeTrainer(
        navigator=navigator,
        problem_cfg=problem_cfg,
        integrator_cfg=cfg["integrator"],
        topology_cfg={**cfg["topology"], **cfg["navigator"]},
        training_cfg=cfg["training"],
        device=device,
    )

    print("Running inference with trajectory logging...")
    traj = trainer.infer_with_trajectory(problem, batches, phi_init)

    phi_hist  = traj["phi_history"]
    w_hist    = traj["weights_history"]
    filt_hist = traj["filtration_history"]
    topo_hist = traj["topo_history"]
    par_hist  = traj["params_history"]
    batches_  = traj["batches"]
    info      = traj["info"]

    T = len(topo_hist)
    print(f"Episode done: T={T} batches, K={K}, D={D}")
    print(f"Final ESS={info['ess_curve'][-1]:.1f},  "
          f"H_W={topo_hist[-1].H_W:.3f},  "
          f"C_tau={topo_hist[-1].C_tau}")

    # -- Generate figures --------------------------------------------------------
    print("\nGenerating figures...")

    if D >= 2:
        print("  Fig 1: data vs estimate")
        fig1_data_vs_estimate(
            problem, phi_hist[-1], w_hist[-1], K, D,
            str(out_dir / "fig1_data_vs_estimate.png"))

        print("  Fig 2: distribution evolution")
        fig2_distribution_evolution(
            problem, phi_hist, w_hist, batches_, K, D,
            str(out_dir / "fig2_distribution_evolution.png"),
            n_snap=args.n_snap)

    print("  Fig 3: mode discovery")
    fig3_mode_discovery(
        problem, phi_hist, w_hist, K, D,
        str(out_dir / "fig3_mode_discovery.png"))

    print("  Fig 4: ESS landscape")
    fig4_ess_landscape(
        topo_hist, info["ess_curve"], w_hist[1:], K,
        str(out_dir / "fig4_ess_landscape.png"))

    if D >= 2:
        print("  Fig 5: IS vs true samples")
        fig5_is_vs_true(
            problem, phi_hist[-1], w_hist[-1], K, D,
            n_samples=500,
            path=str(out_dir / "fig5_is_vs_true_samples.png"))

    print("  Fig 6: navigator parameters")
    fig6_navigator_params(
        par_hist,
        str(out_dir / "fig6_navigator_params.png"))

    print("  Fig 7: phase-space scatter")
    fig7_phase_space(
        phi_hist, w_hist, K, D,
        str(out_dir / "fig7_phase_space.png"),
        n_snap=args.n_snap)

    print("  Fig 8: embedding cluster structure")
    fig8_embedding_clusters(
        phi_hist, w_hist, batches_, K, D,
        str(out_dir / "fig8_embedding_clusters.png"),
        n_snap=4)

    print("  Fig 9: energy + convergence")
    fig9_energy_convergence(
        info["energy_curve"], info["ess_curve"], topo_hist, K,
        str(out_dir / "fig9_energy_convergence.png"))

    print("  Fig 10: PHC topology")
    fig10_phc_topology(
        filt_hist, topo_hist, K,
        str(out_dir / "fig10_phc_topology.png"))

    if D >= 2:
        print("  Fig 11: summary (paper-ready)")
        fig11_summary(
            problem, phi_hist, w_hist, topo_hist, batches_, K, D,
            str(out_dir / "fig11_summary.png"))

    print("  Fig 12: latent cluster growth timeline")
    fig12_cluster_growth_timeline(
        topo_hist, filt_hist, phi_hist, w_hist, K, D,
        str(out_dir / "fig12_cluster_growth_timeline.png"))

    if D >= 2:
        print("  Fig 13: data space vs embedding space snapshots")
        fig13_dataspace_vs_embedding(
            phi_hist, w_hist, filt_hist, batches_, K, D,
            str(out_dir / "fig13_dataspace_vs_embedding.png"),
            n_snap=args.n_snap)

    print(f"\nAll figures saved to  {out_dir}/")
    print("Files:")
    for p in sorted(out_dir.glob("fig*.png")):
        kb = p.stat().st_size // 1024
        print(f"  {p.name}  ({kb} KB)")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 12 — Latent cluster growth: C_tau over time, coloured by input-space mode
# ═════════════════════════════════════════════════════════════════════════════

def fig12_cluster_growth_timeline(
    topo_history: list,
    filtration_history: list,
    phi_history: list,
    weights_history: list,
    K: int,
    D: int,
    path: str,
):
    """
    Two-panel figure showing how clusters grow over streaming batches.

    Panel A (top):  C_tau vs batch t, with per-cluster mass W_c stacked as
                    area chart. Shows "which latent cluster carries how much
                    mass" at every step.

    Panel B (bottom): For each cluster c that is alive at each step t, what
                      GMM component (argmax responsibility) do its member
                      particles predominantly belong to?  Shown as a heatmap:
                      rows = latent clusters (0..C_max-1), cols = time steps,
                      colour = dominant GMM component index.

    This directly answers the two questions:
        - "Does C_tau grow as data arrives?"
        - "Do latent clusters correspond 1-to-1 with input modes?"
    """
    from data.gmm_problem import unpack_phi

    T     = len(topo_history)
    C_max = topo_history[0].C_max if topo_history else 8
    ts    = list(range(1, T + 1))   # batch indices (1-indexed)

    # ── Collect per-(t, c) data ───────────────────────────────────────────────
    # W_c_mat[t, c]      = mass of latent cluster c at step t (0 if absent)
    # dom_comp_mat[t, c] = dominant GMM component of cluster c at step t (-1 if absent)
    W_c_mat      = np.zeros((T, C_max))
    dom_comp_mat = np.full((T, C_max), -1, dtype=int)

    for t_idx, (topo, filt) in enumerate(zip(topo_history, filtration_history)):
        phi_t  = phi_history[t_idx + 1]   # +1: phi_history[0] is init
        w_t    = weights_history[t_idx + 1]
        assign = _np(filt.assignments)    # [M]

        # Per-cluster mass from diagnostics
        for c in range(min(topo.C_tau, C_max)):
            W_c_mat[t_idx, c] = float(topo.W_c[c].item())

        # Dominant GMM component per latent cluster
        # Unpack phi to get pi_tilde → responsibilities in parameter space.
        # We use pi_tilde argmax as a fast proxy: for each particle m, its
        # "favoured component" is argmax softmax(pi_tilde_m).
        # (Responsibility-based version needs X_ref; not available here.)
        try:
            phi_np = _np(phi_t)
            pi_tilde, mu, _ = unpack_phi(phi_t, K, D)
            pi_soft = _np(torch.softmax(pi_tilde, dim=-1))  # [M, K]
            favoured = pi_soft.argmax(axis=1)               # [M]

            for c in range(min(topo.C_tau, C_max)):
                mask_c = (assign == c)
                if mask_c.sum() == 0:
                    continue
                w_c = _np(w_t)[mask_c]
                fav_c = favoured[mask_c]
                # Weighted vote for dominant GMM component
                votes = np.zeros(K)
                for k in range(K):
                    votes[k] = w_c[fav_c == k].sum()
                dom_comp_mat[t_idx, c] = int(votes.argmax())
        except Exception:
            pass

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                             height_ratios=[1.8, 1.4, 1.0])

    # ── Panel A: C_tau and stacked W_c area chart ─────────────────────────────
    ax_c  = fig.add_subplot(gs[0])
    ax_w  = ax_c.twinx()

    c_tau = [t.C_tau for t in topo_history]
    ax_c.step(ts, c_tau, where="mid", color="black", lw=2.0,
              label="$C_\\tau$ (latent clusters)")
    ax_c.set_ylabel("$C_\\tau$ — number of latent clusters", fontsize=10)
    ax_c.set_ylim(0, max(c_tau) + 1.5)
    ax_c.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Stacked area: W_c per cluster
    bottom = np.zeros(T)
    for c in range(C_max):
        wc = W_c_mat[:, c]
        if wc.max() < 1e-6:
            continue
        color = PALETTE[c % len(PALETTE)]
        ax_w.fill_between(ts, bottom, bottom + wc,
                          alpha=0.35, color=color,
                          label=f"Cluster {c} mass")
        bottom = bottom + wc

    ax_w.set_ylabel("Cluster mass $W_c$", fontsize=10, color="gray")
    ax_w.tick_params(axis="y", colors="gray")
    ax_w.set_ylim(0, 1.15)

    ax_c.set_title(
        "Latent cluster count $C_\\tau$ and per-cluster mass over streaming batches\n"
        "Black step = #clusters  |  Shaded areas = cluster masses  "
        "(colour matches component palette)",
        fontweight="bold")
    ax_c.legend(loc="upper left", fontsize=8)
    ax_w.legend(loc="upper right", fontsize=7, ncol=2)
    ax_c.set_xlabel("Streaming batch $t$")

    # ── Panel B: dominant GMM component per latent cluster (heatmap) ──────────
    ax_h = fig.add_subplot(gs[1])

    # Only show rows where cluster was ever alive
    ever_alive = np.any(W_c_mat > 1e-6, axis=0)
    live_rows  = np.where(ever_alive)[0]
    if len(live_rows) == 0:
        live_rows = np.arange(min(C_max, 4))

    hmap = dom_comp_mat[:, live_rows].T.astype(float)  # [n_live, T]
    hmap[hmap < 0] = np.nan

    # Use a discrete colormap with K colours
    cmap_k = matplotlib.colors.ListedColormap(PALETTE[:K])
    bounds = np.arange(-0.5, K + 0.5, 1.0)
    norm_k = matplotlib.colors.BoundaryNorm(bounds, cmap_k.N)

    im = ax_h.imshow(hmap, aspect="auto", cmap=cmap_k, norm=norm_k,
                     origin="lower", extent=[0.5, T + 0.5, -0.5, len(live_rows) - 0.5],
                     interpolation="nearest")

    cb = fig.colorbar(im, ax=ax_h, ticks=range(K), pad=0.02)
    cb.set_label("Dominant GMM component", fontsize=8)
    cb.ax.set_yticklabels([f"Comp {k}" for k in range(K)], fontsize=7)

    ax_h.set_yticks(range(len(live_rows)))
    ax_h.set_yticklabels([f"Cluster {c}" for c in live_rows], fontsize=8)
    ax_h.set_xlabel("Streaming batch $t$")
    ax_h.set_title(
        "Which GMM component does each latent cluster correspond to?\n"
        "Colour = dominant GMM component (weighted argmax $\\pi_k$) of cluster members",
        fontweight="bold")

    # ── Panel C: per-cluster ESS over time ───────────────────────────────────
    ax_e = fig.add_subplot(gs[2])
    for c in live_rows:
        ess_c = [float(t.ESS_c[c].item()) for t in topo_history]
        ax_e.plot(ts, ess_c, color=PALETTE[c % len(PALETTE)],
                  lw=1.5, alpha=0.85, label=f"Cluster {c}")
    global_ess = [t.global_ESS for t in topo_history]
    ax_e.plot(ts, global_ess, "k--", lw=1.5, alpha=0.6, label="Global ESS")
    ax_e.set_xlabel("Streaming batch $t$")
    ax_e.set_ylabel("ESS")
    ax_e.set_title("Per-cluster ESS  (sparse cluster → low ESS → mode B should boost it)",
                   fontweight="bold")
    ax_e.legend(fontsize=7, ncol=min(4, len(live_rows) + 1))

    fig.suptitle(
        "Latent Cluster Structure vs Input Distribution Modes",
        fontsize=13, fontweight="bold", y=0.98)

    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 13 — Side-by-side snapshots: data space vs embedding space
# ═════════════════════════════════════════════════════════════════════════════

def fig13_dataspace_vs_embedding(
    phi_history: list,
    weights_history: list,
    filtration_history: list,
    batches_out: list,
    K: int,
    D: int,
    path: str,
    n_snap: int = 5,
):
    """
    For n_snap timesteps, show two panels side-by-side:

    LEFT  — Data space (x1/x2 projection of the data batch),
             particles overlaid as scatter coloured by their dominant
             GMM component (argmax softmax(pi_tilde_m) for each particle m).
             Size ∝ importance weight.

    RIGHT — Embedding space (PCA of phi), particles coloured by their
             PHC cluster assignment from the filtration at that step.
             Size ∝ importance weight.

    Connecting the two: if latent cluster c ↔ GMM component k, then the
    colour in the right panel should match the colour of the dominant
    GMM component in the left panel for those same particles.
    """
    from data.gmm_problem import unpack_phi

    T     = len(filtration_history)
    snaps = np.linspace(0, T - 1, min(n_snap, T), dtype=int)

    fig, axes = plt.subplots(len(snaps), 2,
                              figsize=(12, 4.2 * len(snaps)))
    if len(snaps) == 1:
        axes = axes[np.newaxis, :]

    cmap_t = cm.viridis
    norm_t = Normalize(vmin=0, vmax=T - 1)

    for row, t_idx in enumerate(snaps):
        ax_data  = axes[row, 0]
        ax_embed = axes[row, 1]

        t_phi    = t_idx + 1            # phi_history[0] is init
        phi_t    = phi_history[t_phi]   # [M, phi_dim]
        w_t      = weights_history[t_phi]  # [M]
        filt     = filtration_history[t_idx]
        bi       = min(t_idx, len(batches_out) - 1)
        X_batch  = batches_out[bi] if bi >= 0 else None

        w_np     = _np(w_t)
        w_norm   = w_np / (w_np.max() + 1e-30)
        sizes    = 18 + 80 * w_norm
        assign   = _np(filt.assignments)   # [M] PHC cluster id, -1=unassigned
        n_clust  = filt.n_clusters

        # ── Compute per-particle dominant GMM component ───────────────────────
        try:
            pi_tilde, mu, _ = unpack_phi(phi_t, K, D)
            pi_soft  = _np(torch.softmax(pi_tilde, dim=-1))  # [M, K]
            dom_comp = pi_soft.argmax(axis=1)                 # [M]
        except Exception:
            dom_comp = np.zeros(phi_t.shape[0], dtype=int)

        comp_colors = [PALETTE[c % len(PALETTE)] for c in dom_comp]

        # ── LEFT: data space (x1/x2 of the batch, particles overlaid) ────────
        if X_batch is not None and D >= 2:
            X_np = _np(X_batch)
            ax_data.scatter(X_np[:, 0], X_np[:, 1],
                            s=6, c="#cccccc", alpha=0.4, zorder=1,
                            label="data batch")

        # For each particle, plot its weighted-mean estimate of mu_k in x1/x2
        # This is more informative than plotting phi directly (phi has K*D dims)
        try:
            mu_np    = _np(mu)      # [M, K, D]
            w_col    = w_np / (w_np.sum() + 1e-30)
            mu_wmean = (w_col[:, None, None] * mu_np).sum(0)  # [K, D] weighted mean

            for k in range(K):
                ax_data.scatter(mu_np[:, k, 0], mu_np[:, k, 1],
                                s=sizes * 0.5, c=PALETTE[k % len(PALETTE)],
                                alpha=0.45, zorder=2, marker="o")
                # Mark weighted-mean estimate
                ax_data.scatter(mu_wmean[k, 0], mu_wmean[k, 1],
                                s=120, c=PALETTE[k % len(PALETTE)],
                                marker="*", zorder=5,
                                edgecolors="black", linewidths=0.6)
        except Exception:
            pass

        ax_data.set_title(
            f"Data space  t={t_idx+1}  |  particles coloured by dominant $\\hat\\mu_k$\n"
            f"★ = IS-weighted mean estimate  ·  dots = all {phi_t.shape[0]} particles",
            fontsize=8, color=cmap_t(norm_t(t_idx)))
        ax_data.set_xlabel("$x_1$"); ax_data.set_ylabel("$x_2$")

        # ── RIGHT: embedding space (PCA of phi) ───────────────────────────────
        try:
            z_np = _np(phi_t)
            try:
                xy, _, _ = _pca2(z_np)
            except Exception:
                xy = z_np[:, :2]

            # Colour by PHC cluster assignment
            clust_colors = []
            for m in range(len(assign)):
                c = int(assign[m])
                clust_colors.append(PALETTE[c % len(PALETTE)] if c >= 0 else "#dddddd")

            ax_embed.scatter(xy[:, 0], xy[:, 1],
                             s=sizes, c=clust_colors,
                             alpha=0.7, edgecolors="none", zorder=2)

            # Annotate cluster centroids
            for c in range(n_clust):
                mask_c = assign == c
                if mask_c.sum() == 0:
                    continue
                cx, cy = xy[mask_c, 0].mean(), xy[mask_c, 1].mean()
                ax_embed.text(cx, cy, str(c), fontsize=9,
                              ha="center", va="center", fontweight="bold",
                              color="white",
                              bbox=dict(boxstyle="round,pad=0.15",
                                        facecolor=PALETTE[c % len(PALETTE)],
                                        alpha=0.85, edgecolor="none"))

        except Exception as e:
            ax_embed.text(0.5, 0.5, f"Error:\n{e}",
                          transform=ax_embed.transAxes,
                          ha="center", va="center", fontsize=8)

        ax_embed.set_title(
            f"Embedding space (PCA of $\\phi$)  t={t_idx+1}\n"
            f"Colour = PHC latent cluster  |  $C_\\tau$ = {n_clust}",
            fontsize=8, color=cmap_t(norm_t(t_idx)))
        ax_embed.set_xlabel("PC1 of $\\phi$")
        ax_embed.set_ylabel("PC2 of $\\phi$")

    # Legend for GMM components (data-space panel)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=PALETTE[k % len(PALETTE)],
                           markersize=8, label=f"GMM comp {k}")
               for k in range(K)]
    fig.legend(handles=handles, loc="lower center",
               ncol=K, fontsize=9,
               title="GMM component (data space colour key)",
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Data Space vs Embedding Space: What Do Latent Clusters Correspond To?\n"
        "Left: particle beliefs about $\\mu_k$ overlaid on data  |  "
        "Right: PHC cluster membership in $\\phi$-space",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    _save(fig, path)


if __name__ == "__main__":
    main()