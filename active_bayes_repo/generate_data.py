from __future__ import annotations

import argparse

from common import load_config, set_seed
from data_generation import generate_active_dataset, save_dataset_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/active_gmm.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    dataset = generate_active_dataset(cfg)
    if bool(cfg.data.save_npz):
        save_dataset_npz(dataset, cfg.data.dataset_path)
        print(f"saved dataset to {cfg.data.dataset_path}")
    else:
        print("config has save_npz=false; dataset generated in memory only")


if __name__ == "__main__":
    main()
