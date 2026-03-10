import itertools, math
import numpy as np

def kmeans_diag(x:np.ndarray, K:int, n_iter:int=16):
    n = len(x)
    centers = x[np.random.choice(n, size=K, replace=n < K)].copy()
    for _ in range(n_iter):
        d = ((x[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        assign = d.argmin(axis=1)
        new = centers.copy()
        for k in range(K):
            idx = np.where(assign == k)[0]
            if len(idx):
                new[k] = x[idx].mean(axis=0)
        if np.allclose(new, centers):
            break
        centers = new
    d = ((x[:, None, :] - centers[None, :, :])**2).sum(axis=2)
    assign = d.argmin(axis=1)
    weights = np.array([(assign == k).mean() for k in range(K)], dtype=np.float32)
    vars_ = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        idx = np.where(assign == k)[0]
        vars_[k] = x[idx].var(axis=0) + 1e-3 if len(idx) else np.array([1.0, 1.0], dtype=np.float32)
    return centers.astype(np.float32), weights, vars_, assign

def best_perm_match(est_centers, est_weights, true_means, true_weights):
    best = None
    for perm in itertools.permutations(range(len(true_means))):
        perm = np.array(perm)
        center_cost = np.sqrt(np.mean(((est_centers - true_means[perm])**2).sum(axis=1)))
        weight_cost = np.abs(est_weights - true_weights[perm]).sum()
        score = center_cost + 0.3 * weight_cost
        if best is None or score < best[0]:
            best = (score, perm, center_cost, weight_cost)
    return best[1], float(best[2]), float(best[3])

def mixture_log_prob(x, weights, means, vars_):
    vals = []
    stds = np.sqrt(vars_)
    for k in range(len(weights)):
        diff = (x - means[k]) / stds[k]
        lp = -0.5 * np.sum(diff**2, axis=1) - np.sum(np.log(stds[k])) - math.log(2 * math.pi)
        vals.append(np.log(weights[k] + 1e-12) + lp)
    vals = np.stack(vals, axis=1)
    m = vals.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(vals - m).sum(axis=1, keepdims=True))).squeeze(1)

def estimate_mode_count(x:np.ndarray, maxK:int=5) -> int:
    centers, weights, _, _ = kmeans_diag(x, maxK, n_iter=10)
    active = [i for i, w in enumerate(weights) if w > 0.08]
    groups = []
    for i in active:
        placed = False
        for g in groups:
            if np.linalg.norm(centers[i] - centers[g[0]]) < 0.9:
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return max(1, len(groups))

def fit_metrics(pred:np.ndarray, gmm, K:int):
    centers, weights, vars_, _ = kmeans_diag(pred, K, n_iter=18)
    _, center_rmse, weight_l1 = best_perm_match(centers, weights, gmm.means, gmm.weights)
    held = gmm.sample(256)
    pred_nll = float(-mixture_log_prob(held, weights, centers, vars_).mean())
    oracle_nll = float(-gmm.log_prob(held).mean())

    a = pred[np.random.choice(len(pred), size=min(len(pred), 80), replace=False)]
    b = gmm.sample(len(a))
    dab = np.sqrt(((a[:, None, :] - b[None, :, :])**2).sum(axis=2))
    daa = np.sqrt(((a[:, None, :] - a[None, :, :])**2).sum(axis=2))
    dbb = np.sqrt(((b[:, None, :] - b[None, :, :])**2).sum(axis=2))
    energy = float(2 * dab.mean() - daa.mean() - dbb.mean())

    khat = estimate_mode_count(pred, maxK=5)
    return {
        'center_rmse': center_rmse,
        'weight_l1': weight_l1,
        'pred_nll': pred_nll,
        'oracle_nll': oracle_nll,
        'energy_dist': energy,
        'mode_count_hat': float(khat),
        'mode_count_acc': 1.0 if khat == K else 0.0,
    }