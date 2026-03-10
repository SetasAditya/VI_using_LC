from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import load_config, set_seed
from data_generation import generate_active_dataset, load_dataset_npz, save_dataset_npz
from evaluation.evaluate import evaluate_checkpoint
from models.assimilator import ActiveLatentAssimilator
from models.projector import FeatureProjector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/active_gmm.yaml")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--policy-mode", type=str, default="learned", choices=["learned","teacher","deficit","coverage_then_deficit","round_robin","random","frontier","centroid_cycle","coverage_guarded","passive_iid","no_query"])
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

    projector = FeatureProjector(int(cfg.data.d_in), int(cfg.data.num_classes), int(cfg.model.hidden_dim),
                                 int(cfg.data.latent_dim), bool(getattr(cfg.model, "identity_encoder", False))).to(device)
    assimilator = ActiveLatentAssimilator(
        int(cfg.data.latent_dim), int(cfg.model.hidden_dim), int(cfg.model.num_particles),
        int(cfg.model.num_clusters), float(cfg.model.obs_sigma), float(cfg.model.teacher_sigma),
        float(cfg.model.cluster_tau), float(cfg.model.rejuvenation), int(cfg.data.num_classes),
        int(getattr(cfg.model, "teacher_move_steps", 6)), float(getattr(cfg.model, "teacher_move_step_size", 0.055)),
        int(getattr(cfg.model, "student_move_steps", 2)), float(getattr(cfg.model, "student_move_step_size", 0.035)),
        float(getattr(cfg.model, "ess_resample_frac", 0.35)),
        float(getattr(cfg.model, "topology_bandwidth", 0.90)),
        float(getattr(cfg.model, "mode_merge_tol", 0.70)),
        float(getattr(cfg.model, "energy_residual_weight", 0.08)),
        float(getattr(cfg.model, "obs_support_weight", 0.35)),
        int(getattr(cfg.model, "num_level_components", max(2 * int(cfg.model.num_clusters), int(cfg.model.num_clusters) + 2))),
        float(getattr(cfg.model, "support_level_blend", 0.35)),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    projector.load_state_dict(ckpt["projector"])
    assimilator.load_state_dict(ckpt["assimilator"])

    metrics, _, path, ablation, ablation_path, ablation_fig, sanity_fig, sanity_paths = evaluate_checkpoint(
        projector, assimilator, dataset, cfg, device, args.split, Path(cfg.output_dir) / "evaluation"
    )
    print(f"saved metrics to {path}")
    print(f"saved policy ablation to {ablation_path}")
    print(f"saved policy ablation figure to {ablation_fig}")
    print(f"saved policy sanity figure to {sanity_fig}")
    for name, sp in sanity_paths.items():
        print(f"sanity/{name}: {sp}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"learned/{k}: {v:.4f}")
    for policy, vals in ablation.items():
        print(f"policy={policy} mean_ari={vals['mean_ari']:.4f} query_acc={vals['query_accuracy']:.4f} regret={vals['mean_query_regret']:.4f} effm={vals['mean_effective_modes']:.4f}")


if __name__ == "__main__":
    main()
