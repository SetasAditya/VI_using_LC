from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .gmm_problem import GMMProblem, sample_gmm_problem


def feature_map_from_latent(z: np.ndarray, labels: np.ndarray, d_in: int, feature_noise: float,
                            rng: np.random.Generator) -> np.ndarray:
    n = z.shape[0]
    z1, z2 = z[:, 0:1], z[:, 1:2]
    poly = np.concatenate([
        z1,
        z2,
        z1 * z2,
        z1 ** 2,
        z2 ** 2,
        np.sin(0.4 * z1),
        np.cos(0.35 * z2),
        np.sin(0.15 * (z1 + z2)),
    ], axis=1).astype(np.float32)

    W = rng.normal(scale=0.55, size=(poly.shape[1], d_in)).astype(np.float32)
    n_classes = int(labels.max()) + 1 if labels.size else 1
    class_emb = rng.normal(scale=0.65, size=(n_classes, d_in)).astype(np.float32)
    x = poly @ W + class_emb[labels]
    if feature_noise > 0:
        x = x + rng.normal(scale=feature_noise, size=(n, d_in)).astype(np.float32)
    return x.astype(np.float32)


def _split_seed(base_seed: int, split_name: str) -> int:
    offsets = {"train": 0, "val": 101, "test": 202}
    return int(base_seed) + offsets.get(split_name, 0)


def _split_overlap(cfg, split_name: str) -> float:
    gmm_cfg = cfg.data.gmm
    key = f"overlap_{split_name}"
    if key in gmm_cfg:
        return float(gmm_cfg[key])
    return float(gmm_cfg.overlap)


def _to_numpy_problem(problem: GMMProblem) -> Dict[str, np.ndarray]:
    return {
        "pi_true": problem.pi_true.detach().cpu().numpy().astype(np.float32),
        "mu_true": problem.mu_true.detach().cpu().numpy().astype(np.float32),
        "Sigma_true": problem.Sigma_true.detach().cpu().numpy().astype(np.float32),
        "L_true": problem.L_true.detach().cpu().numpy().astype(np.float32),
    }


def build_stream_batches(x: np.ndarray, z: np.ndarray, y: np.ndarray, batch_size: int, num_steps: int,
                         rng: np.random.Generator, mode: str = "balanced", shuffle: bool = True,
                         class_probs: Optional[np.ndarray] = None) -> List[Dict[str, np.ndarray]]:
    n = x.shape[0]
    if mode not in {"balanced", "shuffled"}:
        raise ValueError(f"unknown stream mode: {mode}")

    if mode == "shuffled":
        perm = rng.permutation(n) if shuffle else np.arange(n)
        out = []
        for t in range(num_steps):
            s = (t * batch_size) % n
            e = min(s + batch_size, n)
            idx = perm[s:e]
            if idx.shape[0] < batch_size:
                idx = np.concatenate([idx, perm[: batch_size - idx.shape[0]]], axis=0)
            out.append({"x": x[idx], "z": z[idx], "y": y[idx], "t": np.array([t], dtype=np.int64)})
        return out

    k = int(y.max()) + 1 if y.size else 1
    if class_probs is None:
        class_probs = np.ones(k, dtype=np.float64) / max(k, 1)
    else:
        class_probs = np.asarray(class_probs, dtype=np.float64)
        class_probs = class_probs / np.maximum(class_probs.sum(), 1e-12)

    counts = np.floor(batch_size * class_probs).astype(int)
    counts[: batch_size - counts.sum()] += 1

    pools = []
    ptrs = []
    for c in range(k):
        idx = np.where(y == c)[0]
        if shuffle:
            idx = rng.permutation(idx)
        pools.append(idx)
        ptrs.append(0)

    out = []
    for t in range(num_steps):
        batch_idx = []
        for c in range(k):
            want = int(counts[c])
            if want <= 0:
                continue
            idx = pools[c]
            if idx.size == 0:
                continue
            taken = []
            while len(taken) < want:
                remain = idx.size - ptrs[c]
                take_now = min(want - len(taken), remain)
                if take_now > 0:
                    taken.append(idx[ptrs[c]: ptrs[c] + take_now])
                    ptrs[c] += take_now
                if ptrs[c] >= idx.size:
                    ptrs[c] = 0
                    if shuffle:
                        idx = rng.permutation(idx)
                        pools[c] = idx
            batch_idx.extend(np.concatenate(taken, axis=0).tolist())
        batch_idx = np.asarray(batch_idx, dtype=np.int64)
        if batch_idx.shape[0] < batch_size:
            extra = rng.choice(np.arange(n), size=batch_size - batch_idx.shape[0], replace=True)
            batch_idx = np.concatenate([batch_idx, extra], axis=0)
        if shuffle:
            rng.shuffle(batch_idx)
        out.append({"x": x[batch_idx], "z": z[batch_idx], "y": y[batch_idx], "t": np.array([t], dtype=np.int64)})
    return out


def _sample_split(cfg, split_name: str) -> Dict[str, np.ndarray]:
    data_cfg = cfg.data
    gmm_cfg = cfg.data.gmm
    split_seed = _split_seed(int(cfg.seed), split_name)
    np_rng = np.random.default_rng(split_seed)
    torch.manual_seed(split_seed)

    problem = sample_gmm_problem(
        K=int(data_cfg.num_classes),
        D=int(data_cfg.latent_dim),
        N=int(getattr(data_cfg, f"total_{split_name}")),
        overlap=_split_overlap(cfg, split_name),
        sigma_scale=float(gmm_cfg.sigma_scale),
        device=torch.device("cpu"),
        dirichlet_alpha=float(gmm_cfg.dirichlet_alpha),
    )

    z = problem.X.detach().cpu().numpy().astype(np.float32)
    latent_noise = float(data_cfg.latent_noise)
    if latent_noise > 0:
        z = z + np_rng.normal(scale=latent_noise, size=z.shape).astype(np.float32)
    y = problem.labels_true.detach().cpu().numpy().astype(np.int64)
    x = feature_map_from_latent(z, y, int(data_cfg.d_in), float(data_cfg.feature_noise), np_rng)

    stream = build_stream_batches(
        x=x,
        z=z,
        y=y,
        batch_size=int(data_cfg.stream_batch_size),
        num_steps=int(data_cfg.num_stream_steps),
        rng=np_rng,
        mode=str(gmm_cfg.stream_mode),
        shuffle=bool(gmm_cfg.stream_shuffle),
        class_probs=problem.pi_true.detach().cpu().numpy().astype(np.float32),
    )

    meta = _to_numpy_problem(problem)
    meta.update({
        "overlap": np.array([_split_overlap(cfg, split_name)], dtype=np.float32),
        "stream_mode": np.array([str(gmm_cfg.stream_mode)]),
    })
    return {
        "x": x,
        "z": z,
        "y": y,
        "stream": stream,
        "meta": meta,
    }


def generate_toy_dataset(cfg) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        "train": _sample_split(cfg, "train"),
        "val": _sample_split(cfg, "val"),
        "test": _sample_split(cfg, "test"),
        "meta": {
            "generator": np.array(["gmm_problem"], dtype=object),
            "num_classes": np.array([int(cfg.data.num_classes)], dtype=np.int64),
            "latent_dim": np.array([int(cfg.data.latent_dim)], dtype=np.int64),
        },
    }


def save_dataset_npz(dataset: Dict[str, Dict[str, np.ndarray]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for split in ["train", "val", "test"]:
        payload[f"{split}_x"] = dataset[split]["x"]
        payload[f"{split}_z"] = dataset[split]["z"]
        payload[f"{split}_y"] = dataset[split]["y"]
        meta = dataset[split].get("meta", {})
        for k, v in meta.items():
            payload[f"{split}_{k}"] = v
    for k, v in dataset.get("meta", {}).items():
        payload[f"meta_{k}"] = v
    np.savez(path, **payload)


def load_dataset_npz(path: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    arr = np.load(path, allow_pickle=True)
    out = {}
    for split in ["train", "val", "test"]:
        meta = {}
        prefix = f"{split}_"
        for key in arr.files:
            if key.startswith(prefix) and key not in {f"{split}_x", f"{split}_z", f"{split}_y"}:
                meta[key[len(prefix):]] = arr[key]
        out[split] = {"x": arr[f"{split}_x"], "z": arr[f"{split}_z"], "y": arr[f"{split}_y"], "meta": meta}
    meta = {}
    for key in arr.files:
        if key.startswith("meta_"):
            meta[key[len("meta_"):]] = arr[key]
    out["meta"] = meta
    return out


def as_torch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray) and v.dtype.kind in ("i", "u"):
            out[k] = torch.as_tensor(v, dtype=torch.long, device=device)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return out
