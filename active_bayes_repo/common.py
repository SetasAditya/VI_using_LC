from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


class AttrDict(dict):
    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path: str | os.PathLike) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        return AttrDict(yaml.safe_load(f))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(path: str | os.PathLike, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(payload, path)
    except RuntimeError:
        torch.save(payload, path, _use_new_zipfile_serialization=False)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj
