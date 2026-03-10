from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _pca_project(x: np.ndarray, k: int = 2) -> np.ndarray:
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    return (x0 @ vt[:k].T).astype(np.float32)


def _feature_map(z: np.ndarray, d_in: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    if d_in == z.shape[1]:
        x = z.copy()
    else:
        feats = [z]
        if z.shape[1] >= 2:
            feats += [
                np.stack([z[:, 0] * z[:, 1], z[:, 0] ** 2 - z[:, 1] ** 2], axis=1),
                np.stack([np.sin(0.8 * z[:, 0]), np.cos(0.8 * z[:, 1])], axis=1),
                np.stack([np.tanh(z[:, 0]), np.tanh(z[:, 1])], axis=1),
            ]
        x = np.concatenate(feats, axis=1)
        if x.shape[1] < d_in:
            W = rng.normal(scale=0.8, size=(x.shape[1], d_in - x.shape[1])).astype(np.float32)
            x = np.concatenate([x, np.tanh(x @ W)], axis=1)
        x = x[:, :d_in]
    x = x + noise * rng.normal(size=x.shape).astype(np.float32)
    return x.astype(np.float32)


def _sample_gmm_meta(num_classes: int, overlap: float, sigma_scale: float, dirichlet_alpha: float,
                     rng: np.random.Generator) -> Dict[str, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, num_classes, endpoint=False)
    radius = 3.6 + 0.9 * rng.random(num_classes)
    means = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1).astype(np.float32)
    means += 0.25 * rng.normal(size=means.shape).astype(np.float32)
    covs = []
    for _ in range(num_classes):
        base = np.diag(np.array([0.12 + 0.8 * overlap, 0.06 + 0.5 * overlap], dtype=np.float32)) * sigma_scale
        rot = _rotation(float(rng.uniform(0, np.pi)))
        covs.append((rot @ base @ rot.T).astype(np.float32))
    covs = np.stack(covs, axis=0)
    pi = rng.dirichlet(alpha=np.full(num_classes, dirichlet_alpha, dtype=np.float32)).astype(np.float32)
    return {"mu_true": means.astype(np.float32), "Sigma_true": covs.astype(np.float32), "pi_true": pi.astype(np.float32)}


def _sample_from_meta(n: int, meta: Dict[str, np.ndarray], d_in: int, feature_noise: float, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    means, covs, pi = meta["mu_true"], meta["Sigma_true"], meta["pi_true"]
    num_classes = means.shape[0]
    y = rng.choice(num_classes, size=n, p=pi)
    z = np.zeros((n, 2), dtype=np.float32)
    for k in range(num_classes):
        idx = np.where(y == k)[0]
        if idx.size:
            z[idx] = rng.multivariate_normal(mean=means[k], cov=covs[k], size=idx.size).astype(np.float32)
    x = _feature_map(z, d_in=d_in, noise=feature_noise, rng=rng)
    return {"x": x.astype(np.float32), "y": y.astype(np.int64), "z": z.astype(np.float32), "meta": meta}


def _sample_gmm_split(n: int, num_classes: int, overlap: float, sigma_scale: float, dirichlet_alpha: float,
                      d_in: int, feature_noise: float, rng: np.random.Generator, meta_override: Dict[str, np.ndarray] | None = None) -> Dict[str, np.ndarray]:
    meta = _sample_gmm_meta(num_classes, overlap, sigma_scale, dirichlet_alpha, rng) if meta_override is None else meta_override
    return _sample_from_meta(n, meta, d_in, feature_noise, rng)


def _localized_from_anchor(z: np.ndarray, anchor: np.ndarray, batch_size: int) -> np.ndarray:
    d2 = ((z - anchor[None, :]) ** 2).sum(axis=1)
    idx = np.argsort(d2)[:batch_size]
    return idx.astype(np.int64)


def _heterogeneous_action_indices(z: np.ndarray, y: np.ndarray, means: np.ndarray, covs: np.ndarray,
                                  num_candidates: int, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    K = means.shape[0]
    anchors = []
    for c in range(min(K, num_candidates)):
        anchors.append(means[c] + rng.normal(scale=0.35, size=2).astype(np.float32))
    while len(anchors) < num_candidates:
        mode = int(rng.integers(0, K))
        if len(anchors) % 3 == 0:
            other = int(rng.integers(0, K))
            if other == mode:
                other = (other + 1) % K
            lam = float(rng.uniform(0.35, 0.65))
            anchor = lam * means[mode] + (1.0 - lam) * means[other]
            anchor = anchor + rng.normal(scale=0.45, size=2).astype(np.float32)
        elif len(anchors) % 3 == 1:
            eigval, eigvec = np.linalg.eigh(covs[mode])
            direction = eigvec[:, np.argmax(eigval)]
            anchor = means[mode] + float(rng.uniform(1.0, 2.3)) * direction.astype(np.float32)
        else:
            anchor = means.mean(axis=0) + rng.normal(scale=2.0, size=2).astype(np.float32)
        anchors.append(anchor.astype(np.float32))
    anchors = np.stack(anchors[:num_candidates], axis=0).astype(np.float32)
    idx_list = []
    for a in anchors:
        base_idx = _localized_from_anchor(z, a, max(batch_size - batch_size // 4, 1))
        far_idx = np.argsort(-((z - a[None, :]) ** 2).sum(axis=1))[:batch_size // 4]
        idx = np.concatenate([base_idx, far_idx], axis=0)
        if idx.shape[0] < batch_size:
            pad = rng.choice(len(z), size=batch_size - idx.shape[0], replace=True)
            idx = np.concatenate([idx, pad])
        rng.shuffle(idx)
        idx_list.append(idx[:batch_size])
    return np.stack(idx_list, axis=0).astype(np.int64), anchors



def _build_passive_episode(split: Dict[str, np.ndarray], num_steps: int, batch_size: int,
                           rng: np.random.Generator) -> Dict[str, np.ndarray]:
    x, y, z = split["x"], split["y"], split["z"]
    N = x.shape[0]
    idx = rng.integers(0, N, size=(num_steps, batch_size), endpoint=False)
    x_cands = x[idx][:, None, :, :].astype(np.float32)
    y_cands = y[idx][:, None, :].astype(np.int64)
    z_cands = z[idx][:, None, :, :].astype(np.float32)
    anchors = z[idx].mean(axis=1, keepdims=True).astype(np.float32)
    target_freq = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float32)
    target_freq /= max(target_freq.sum(), 1.0)
    target_centroids = np.stack([z[y == c].mean(axis=0) if np.any(y == c) else np.zeros(z.shape[1]) for c in range(len(target_freq))], axis=0).astype(np.float32)
    return {**split, "candidate_x": x_cands, "candidate_y": y_cands, "candidate_z": z_cands,
            "candidate_anchor": anchors, "target_freq": target_freq, "target_centroids": target_centroids}


def _build_episode(split: Dict[str, np.ndarray], num_steps: int, num_candidates: int, batch_size: int,
                   rng: np.random.Generator, balanced: bool = False) -> Dict[str, np.ndarray]:
    x, y, z = split["x"], split["y"], split["z"]
    means = split.get("meta", {}).get("mu_true", np.stack([z[y == c].mean(axis=0) for c in np.unique(y)], axis=0).astype(np.float32))
    covs = split.get("meta", {}).get("Sigma_true", np.stack([np.cov(z[y == c].T) + 1e-3 * np.eye(z.shape[1]) for c in np.unique(y)], axis=0).astype(np.float32))
    C, B = num_candidates, batch_size
    x_cands = np.zeros((num_steps, C, B, x.shape[1]), dtype=np.float32)
    y_cands = np.zeros((num_steps, C, B), dtype=np.int64)
    z_cands = np.zeros((num_steps, C, B, z.shape[1]), dtype=np.float32)
    anchors = np.zeros((num_steps, C, z.shape[1]), dtype=np.float32)
    for t in range(num_steps):
        idx, anchors_t = _heterogeneous_action_indices(z, y, means, covs, C, B, rng)
        x_cands[t] = x[idx]
        y_cands[t] = y[idx]
        z_cands[t] = z[idx]
        anchors[t] = anchors_t
    target_freq = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float32)
    target_freq /= max(target_freq.sum(), 1.0)
    target_centroids = np.stack([z[y == c].mean(axis=0) if np.any(y == c) else np.zeros(z.shape[1]) for c in range(len(target_freq))], axis=0).astype(np.float32)
    return {**split, "candidate_x": x_cands, "candidate_y": y_cands, "candidate_z": z_cands, "candidate_anchor": anchors, "target_freq": target_freq, "target_centroids": target_centroids}


def _load_external_npz(path: str | Path, latent_dim: int, num_stream_steps: int, num_candidates: int, batch_size: int,
                       rng: np.random.Generator) -> Dict[str, Dict[str, np.ndarray]]:
    arr = np.load(path, allow_pickle=True)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        x = arr[f"x_{split}"].astype(np.float32)
        y = arr[f"y_{split}"].astype(np.int64)
        z_key = f"z_{split}"
        z = arr[z_key].astype(np.float32) if z_key in arr else _pca_project(x, k=latent_dim)
        split_dict = {"x": x, "y": y, "z": z, "meta": {}}
        out[split] = _build_episode(split_dict, num_stream_steps, num_candidates, batch_size, rng, balanced=False)
    return out


def generate_active_dataset(cfg) -> Dict[str, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(cfg.seed))
    query_scheme = str(getattr(cfg.data, "query_scheme", "active_localized"))
    if str(cfg.data.source) == "labeled_npz":
        return _load_external_npz(cfg.data.external_npz_path, latent_dim=int(cfg.data.latent_dim), num_stream_steps=int(cfg.data.num_stream_steps), num_candidates=int(cfg.data.num_candidates), batch_size=int(cfg.data.query_batch_size), rng=rng)

    dataset = {}
    share_params = bool(getattr(cfg.data.gmm, "share_params", False))
    shared_meta = None
    if share_params:
        shared_meta = _sample_gmm_meta(
            num_classes=int(cfg.data.num_classes),
            overlap=float(cfg.data.gmm.overlap_train),
            sigma_scale=float(cfg.data.gmm.sigma_scale),
            dirichlet_alpha=float(cfg.data.gmm.dirichlet_alpha),
            rng=np.random.default_rng(int(cfg.seed) + 999),
        )
    for split_name, n, overlap in [
        ("train", int(cfg.data.total_train), float(cfg.data.gmm.overlap_train)),
        ("val", int(cfg.data.total_val), float(cfg.data.gmm.overlap_val)),
        ("test", int(cfg.data.total_test), float(cfg.data.gmm.overlap_test)),
    ]:
        split_rng = np.random.default_rng(int(cfg.seed) + (sum(map(ord, split_name)) % 10000))
        split = _sample_gmm_split(
            n=n,
            num_classes=int(cfg.data.num_classes),
            overlap=overlap,
            sigma_scale=float(cfg.data.gmm.sigma_scale),
            dirichlet_alpha=float(cfg.data.gmm.dirichlet_alpha),
            d_in=int(cfg.data.d_in),
            feature_noise=float(cfg.data.feature_noise),
            rng=split_rng,
            meta_override=shared_meta,
        )
        episode_rng = np.random.default_rng(int(cfg.seed) + 100 + (sum(map(ord, split_name)) % 10000))
        if query_scheme == "passive_iid":
            dataset[split_name] = _build_passive_episode(split, num_steps=int(cfg.data.num_stream_steps), batch_size=int(cfg.data.query_batch_size), rng=episode_rng)
        else:
            dataset[split_name] = _build_episode(split, num_steps=int(cfg.data.num_stream_steps), num_candidates=int(cfg.data.num_candidates), batch_size=int(cfg.data.query_batch_size), rng=episode_rng, balanced=False)
    return dataset


def save_dataset_npz(dataset: Dict[str, Dict[str, np.ndarray]], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for split, d in dataset.items():
        for k, v in d.items():
            if isinstance(v, dict):
                payload[f"{split}_{k}"] = np.array(v, dtype=object)
            else:
                payload[f"{split}_{k}"] = v
    np.savez(path, **payload)


def load_dataset_npz(path: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    arr = np.load(path, allow_pickle=True)
    out: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "val": {}, "test": {}}
    for k in arr.files:
        split, rest = k.split("_", 1)
        if arr[k].dtype == object and arr[k].shape == ():
            out[split][rest] = arr[k].item()
        else:
            out[split][rest] = arr[k]
    return out
