"""
Evaluate trained GMM SPHS Navigator.

Runs inference on held-out test problems and computes:
    - MSE_mu, MSE_Sigma (parameter accuracy)
    - MMD to reference posterior
    - Mode discovery rate, time to discover K
    - ESS efficiency
    - Comparison vs baselines (plain BAOAB, EM)

Usage:
    python scripts/evaluate.py --checkpoint outputs/navigator_best.pt
                                --config configs/gmm.yaml
                                --n_test 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import numpy as np

from data.gmm_problem import (
    GMMProblem, phi_dim as compute_phi_dim, sample_gmm_problem,
    get_streaming_batches, sample_prior_particles, unpack_phi
)
from dynamics.gmm_energy import GMMEnergy
from dynamics.canonicalize import hungarian_match
from data.gmm_problem import L_vec_to_matrix


def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> float:
    """MMD² between two sample sets."""
    n, m = x.shape[0], y.shape[0]
    def k(a, b):
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return torch.exp(-(diff**2).sum(-1) / (2*sigma**2))
    Kxx = k(x, x)
    Kyy = k(y, y)
    Kxy = k(x, y)
    return (
        (Kxx.sum() - Kxx.diag().sum()) / (n*(n-1)) +
        (Kyy.sum() - Kyy.diag().sum()) / (m*(m-1)) -
        2 * Kxy.mean()
    ).item()


def _component_coverage(
    mu: torch.Tensor,           # [M, K, D] per-particle mean estimates
    weights: torch.Tensor,      # [M] normalized weights
    mu_true: torch.Tensor,      # [K, D] true means
    Sigma_true: torch.Tensor,   # [K, D, D] true covariances
    K: int,
    D: int,
    coverage_sigma: float = 1.5,  # component covered if weighted-mean estimate within
                                   # coverage_sigma * sqrt(max_eigenvalue(Sigma_k))
) -> tuple:
    """
    Ground-truth-based mode coverage metric.

    A component k is "covered" if the WEIGHTED-MEAN estimate for that component
    is within coverage_sigma * sigma_k of the true mean, where sigma_k is the
    largest standard deviation of component k (sqrt of max eigenvalue of Sigma_k).

    Using weighted-mean (not best particle) because:
    - Best-particle metric inflates coverage: a stray particle landing near a
      true component registers as "covered" even if 127 others missed it.
    - Weighted mean represents where the particle cloud actually believes mu_k is.
    - A threshold of 1.5 sigma is a strict test: the estimate must be inside the
      component's 1.5-sigma ellipse.

    Returns:
        n_covered: int — number of true components covered
        coverage_rate: float in [0,1]
        per_component_dist_in_sigma: [K] weighted-mean error in units of sigma_k
        weighted_mean_dist: [K] raw L2 distance of weighted mean to true mean
    """
    # Weighted-mean estimate per component
    mu_w = (weights.unsqueeze(-1).unsqueeze(-1) * mu).sum(0)  # [K, D]

    # Hungarian-match estimated means to true means
    try:
        perm = hungarian_match(mu_w, mu_true)
        mu_w_matched = mu_w[perm]                  # [K, D]
    except Exception:
        perm = torch.arange(K, device=mu.device)
        mu_w_matched = mu_w

    # Per-component raw L2 error
    raw_dist = ((mu_w_matched - mu_true) ** 2).sum(-1).sqrt()  # [K]

    # Per-component threshold: coverage_sigma * largest std of that component
    # sigma_k = sqrt(max eigenvalue of Sigma_k)
    # For a 2D Gaussian, this is the semi-major axis length
    sigma_k = torch.zeros(K, device=mu_true.device)
    for k in range(K):
        try:
            eigvals = torch.linalg.eigvalsh(Sigma_true[k])    # ascending eigenvalues
            sigma_k[k] = eigvals.max().clamp(min=1e-4).sqrt()
        except Exception:
            sigma_k[k] = 1.0

    threshold_k = coverage_sigma * sigma_k                    # [K] per-component threshold

    # Error in units of sigma_k
    dist_in_sigma = raw_dist / sigma_k.clamp(min=1e-6)       # [K]

    covered = (raw_dist < threshold_k)                        # [K] bool
    n_covered = covered.sum().item()

    return int(n_covered), n_covered / K, dist_in_sigma.detach(), raw_dist.detach()


def evaluate_one_problem(
    navigator,
    problem: GMMProblem,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Run inference and compute all metrics for one problem."""
    from training.episode_trainer import EpisodeTrainer

    K = problem.K
    D = problem.D
    n_particles = cfg["particle"]["n_particles"]
    T_batches = cfg["problem"]["T_batches"]
    batch_size = cfg["problem"]["batch_size"]

    batches = get_streaming_batches(problem, T_batches, batch_size)
    phi_init = sample_prior_particles(n_particles, K, D, problem, device=device)

    problem_cfg = cfg["problem"].copy()
    problem_cfg["n_particles"] = n_particles

    trainer = EpisodeTrainer(
        navigator=navigator,
        problem_cfg=problem_cfg,
        integrator_cfg=cfg["integrator"],
        topology_cfg={**cfg["topology"], **cfg["navigator"]},
        training_cfg=cfg["training"],
        device=device,
    )

    result = trainer.infer(problem, batches, phi_init)

    phi_final = result["phi"]     # [M, phi_dim]
    weights   = result["weights"] # [M]
    info      = result["info"]

    pi_tilde, mu, L_vecs = unpack_phi(phi_final, K, D)

    # Weighted mean estimates
    mu_weighted = (weights.unsqueeze(-1).unsqueeze(-1) * mu).sum(0)   # [K, D]
    L = L_vec_to_matrix(L_vecs, D)
    Sigma = torch.bmm(L.reshape(-1, D, D), L.reshape(-1, D, D).transpose(-1,-2))
    Sigma = Sigma.reshape(phi_final.shape[0], K, D, D)
    Sigma_weighted = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * Sigma).sum(0)

    # ── Parameter accuracy (Hungarian matched) ───────────────────────────────
    try:
        perm = hungarian_match(mu_weighted, problem.mu_true)
        # Apply perm to PREDICTION, not to ground truth
        mse_mu    = ((mu_weighted[perm] - problem.mu_true) ** 2).mean().item()
        mse_sigma = ((Sigma_weighted[perm] - problem.Sigma_true) ** 2).mean().item()
    except Exception:
        mse_mu    = float("nan")
        mse_sigma = float("nan")

    # ── Mode coverage (ground-truth aware, replaces C_tau-based MER) ─────────
    n_covered, coverage_rate, dist_in_sigma, raw_dist = _component_coverage(
        mu=mu, weights=weights,
        mu_true=problem.mu_true, Sigma_true=problem.Sigma_true,
        K=K, D=D,
    )

    # ── time_to_coverage: first batch where all components are within threshold ─
    # We only have terminal phi here, so compute a proxy: batch where
    # energy drops below 110% of final energy (proxy for "converged").
    energy_hist = info["energy_curve"]
    if energy_hist:
        final_E = energy_hist[-1]
        converge_batch = next(
            (t + 1 for t, e in enumerate(energy_hist) if e <= final_E * 1.1),
            len(energy_hist),
        )
    else:
        converge_batch = None

    # ── ESS ──────────────────────────────────────────────────────────────────
    ess_final = info["ess_curve"][-1] if info["ess_curve"] else 0.0

    # ── MMD between weighted particle mu distribution and true mu ─────────────
    mu_flat      = mu.reshape(phi_final.shape[0], -1)          # [M, K*D]
    mu_true_flat = problem.mu_true.flatten().unsqueeze(0).expand(phi_final.shape[0], -1)
    try:
        sigma_mmd = mu_flat.std().item() + 0.1
        mmd = mmd_loss(mu_flat, mu_true_flat, sigma=sigma_mmd)
    except Exception:
        mmd = float("nan")

    # ── PHC diagnostics (kept for reference, not used in primary metrics) ─────
    C_history = info["C_tau_curve"]
    C_final   = C_history[-1] if C_history else 0

    return {
        "K": K,
        # Primary parameter metrics
        "mse_mu":    mse_mu,
        "mse_sigma": mse_sigma,
        "mmd":       mmd,
        # Coverage metrics (ground-truth based)
        "n_covered":          n_covered,
        "coverage_rate":      coverage_rate,        # fraction of K components found
        "dist_in_sigma":      dist_in_sigma.tolist(),  # per-component error in units of sigma_k
        "raw_dist_to_true":   raw_dist.tolist(),    # per-component L2 to true mean
        "converge_batch":     converge_batch,       # replaces time_to_K
        # Sampling quality
        "ess_final": ess_final,
        # PHC reference (informational only — C=1 is expected at convergence)
        "C_final":   C_final,
        "C_history": C_history,
        "ess_history":    info["ess_curve"],
        "energy_history": info["energy_curve"],
    }


def evaluate_em_baseline(problem: GMMProblem, n_iter: int = 100) -> dict:
    """Simple EM baseline."""
    from sklearn.mixture import GaussianMixture
    import numpy as np

    X_np = problem.X.cpu().numpy()
    gm = GaussianMixture(n_components=problem.K, max_iter=n_iter, n_init=3)
    gm.fit(X_np)

    mu_pred = torch.tensor(gm.means_, dtype=torch.float32)
    mu_true = problem.mu_true.cpu()

    try:
        perm = hungarian_match(mu_pred, mu_true)
        mse_mu = ((mu_pred[perm.cpu()] - mu_true)**2).mean().item()
    except:
        mse_mu = float("nan")

    return {"mse_mu": mse_mu, "method": "EM"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/gmm.yaml")
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="outputs/eval_results.json")
    parser.add_argument("--K_fixed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device or cfg.get("device", "cpu"))

    # Load navigator — use dimensions from checkpoint, not YAML.
    # The checkpoint may have been trained with a fixed K (--K flag) or
    # a different K_max than the current config.
    from scripts.train import build_navigator

    D = cfg["problem"]["D"]
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Prefer stored dimensions; fall back to recomputing from config.
    ckpt_cfg = ckpt.get("config", {})
    feat_dim   = ckpt_cfg.get("feat_dim",  None)
    phi_dim_ck = ckpt_cfg.get("phi_dim",   None)

    if feat_dim is None or phi_dim_ck is None:
        from scripts.train import compute_feat_dim
        from data.gmm_problem import phi_dim as compute_phi_dim
        K_max = cfg["problem"]["K_max"]
        feat_dim   = compute_feat_dim(cfg)
        phi_dim_ck = compute_phi_dim(K_max, D)
        print(f"[warn] checkpoint has no stored dims; recomputing feat_dim={feat_dim}, phi_dim={phi_dim_ck}")
    else:
        print(f"Using checkpoint dims: feat_dim={feat_dim}, phi_dim={phi_dim_ck}")

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

    print(f"Loaded navigator from {args.checkpoint}")
    print(f"Running {args.n_test} test problems...")

    results = []
    em_results = []

    # Determine max K the navigator can handle (its phi_dim determines max K).
    # phi_dim = K + K*D + K*D*(D+1)//2  =>  K_nav_max from stored phi_dim_ck
    chol_sz = D * (D + 1) // 2
    # phi_dim = K*(1 + D + chol_sz)  =>  K_nav_max = phi_dim // (1+D+chol_sz)
    K_nav_max = phi_dim_ck // (1 + D + chol_sz)
    K_eval_min = cfg["problem"]["K_min"]
    K_eval_max = min(cfg["problem"]["K_max"], K_nav_max)
    print(f"Evaluating K in [{K_eval_min}, {K_eval_max}] (navigator phi_dim={phi_dim_ck} => K_nav_max={K_nav_max})")

    for i in range(args.n_test):
        import random
        K = args.K_fixed or random.randint(K_eval_min, K_eval_max)

        problem = sample_gmm_problem(
            K=K, D=D,
            N=cfg["problem"]["N"],
            overlap=0.3,
            device=device,
        )

        try:
            r = evaluate_one_problem(navigator, problem, cfg, device)
            results.append(r)
        except Exception as e:
            print(f"  Problem {i} failed: {e}")
            continue

        try:
            em_r = evaluate_em_baseline(problem)
            em_results.append(em_r)
        except:
            pass

        if (i + 1) % 10 == 0:
            valid = [r for r in results if not np.isnan(r["mse_mu"])]
            if valid:
                avg_mse      = np.mean([r["mse_mu"] for r in valid])
                avg_coverage = np.mean([r["coverage_rate"] for r in valid])
                print(f"  {i+1}/{args.n_test} | avg_mse_mu={avg_mse:.4f} | avg_coverage={avg_coverage:.3f}")

    # Summarize
    valid = [r for r in results if not np.isnan(r.get("mse_mu", float("nan")))]
    print(f"\n{'='*60}")
    print(f"RESULTS  ({len(valid)}/{args.n_test} valid)")
    print(f"{'='*60}")

    for metric in ["mse_mu", "mse_sigma", "ess_final", "coverage_rate"]:
        vals = [r[metric] for r in valid if not np.isnan(r.get(metric, float("nan")))]
        if vals:
            print(f"  {metric:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Convergence speed
    conv = [r["converge_batch"] for r in valid if r.get("converge_batch") is not None]
    if conv:
        print(f"  {'converge_batch':25s}: {np.mean(conv):.1f} ± {np.std(conv):.1f} batches")

    # Per-component breakdown (in sigma units — 1.0 = exactly 1 std away)
    all_dist_sigma = [r["dist_in_sigma"] for r in valid if r.get("dist_in_sigma")]
    if all_dist_sigma:
        flat = [d for row in all_dist_sigma for d in row]
        print(f"  {'mean err (sigma units)':25s}: {np.mean(flat):.3f} ± {np.std(flat):.3f}  (threshold=1.5)")
    all_raw = [r["raw_dist_to_true"] for r in valid if r.get("raw_dist_to_true")]
    if all_raw:
        flat = [d for row in all_raw for d in row]
        print(f"  {'mean L2 to true mu':25s}: {np.mean(flat):.4f} ± {np.std(flat):.4f}")

    # Proportion of failures (coverage_rate < 0.5)
    failures = [r for r in valid if r.get("coverage_rate", 1.0) < 0.5]
    print(f"  {'coverage<50% (failures)':25s}: {len(failures)}/{len(valid)} ({100*len(failures)/max(len(valid),1):.1f}%)")

    if em_results:
        em_mse = np.mean([r["mse_mu"] for r in em_results if not np.isnan(r["mse_mu"])])
        print(f"\nEM baseline mse_mu: {em_mse:.4f}")

    # Save
    out = {"navigator": valid, "em_baseline": em_results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        # Convert tensors to floats for JSON
        def convert(obj):
            if isinstance(obj, (torch.Tensor,)):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            return obj
        import json
        json.dump(out, f, default=str, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()