from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import ensure_dir, load_config, save_checkpoint, set_seed, to_plain
from data_generation import generate_active_dataset, load_dataset_npz, save_dataset_npz
from models.assimilator import ActiveLatentAssimilator
from models.projector import FeatureProjector
from training.trainer import train_assimilator, train_projector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/active_gmm.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() or str(cfg.device) == "cpu" else "cpu")

    if bool(cfg.data.save_npz) and Path(cfg.data.dataset_path).exists():
        dataset = load_dataset_npz(cfg.data.dataset_path)
    else:
        dataset = generate_active_dataset(cfg)
        if bool(cfg.data.save_npz):
            save_dataset_npz(dataset, cfg.data.dataset_path)

    output_dir = ensure_dir(cfg.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")

    projector = FeatureProjector(
        d_in=int(cfg.data.d_in),
        num_classes=int(cfg.data.num_classes),
        hidden_dim=int(cfg.model.hidden_dim),
        latent_dim=int(cfg.data.latent_dim),
        identity_encoder=bool(getattr(cfg.model, "identity_encoder", False)),
    ).to(device)
    assimilator = ActiveLatentAssimilator(
        latent_dim=int(cfg.data.latent_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_particles=int(cfg.model.num_particles),
        num_clusters=int(cfg.model.num_clusters),
        obs_sigma=float(cfg.model.obs_sigma),
        teacher_sigma=float(cfg.model.teacher_sigma),
        cluster_tau=float(cfg.model.cluster_tau),
        rejuvenation=float(cfg.model.rejuvenation),
        num_classes=int(cfg.data.num_classes),
        teacher_move_steps=int(getattr(cfg.model, "teacher_move_steps", 6)),
        teacher_move_step_size=float(getattr(cfg.model, "teacher_move_step_size", 0.055)),
        student_move_steps=int(getattr(cfg.model, "student_move_steps", 2)),
        student_move_step_size=float(getattr(cfg.model, "student_move_step_size", 0.035)),
        ess_resample_frac=float(getattr(cfg.model, "ess_resample_frac", 0.35)),
        topology_bandwidth=float(getattr(cfg.model, "topology_bandwidth", 0.90)),
        mode_merge_tol=float(getattr(cfg.model, "mode_merge_tol", 0.70)),
        energy_residual_weight=float(getattr(cfg.model, "energy_residual_weight", 0.08)),
        obs_support_weight=float(getattr(cfg.model, "obs_support_weight", 0.35)),
    ).to(device)

    proj_stats = train_projector(projector, cfg, dataset, device, output_dir)
    assim_stats = train_assimilator(projector, assimilator, cfg, dataset, device, output_dir)

    save_checkpoint(ckpt_dir / "best.pt", {
        "projector": projector.state_dict(),
        "assimilator": assimilator.state_dict(),
        "config": to_plain(cfg),
        "projector_stats": proj_stats,
        "assimilator_stats": assim_stats,
    })
    with open(log_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump({"projector": proj_stats, "assimilator": assim_stats}, f, indent=2)
    print(f"saved checkpoint to {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
