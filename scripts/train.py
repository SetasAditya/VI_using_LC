"""
Train GMM SPHS Navigator.

Usage:
    python scripts/train.py --config configs/gmm.yaml --output outputs/
    python scripts/train.py --config configs/gmm.yaml --K 3 --n_episodes 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from data.gmm_problem import phi_dim as compute_phi_dim
from models.gmm_navigator import GMMNavigator
from topology.phc import PHC
from topology.diagnostics import TopoDiagnostics
from training.episode_trainer import EpisodeTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_feat_dim(cfg: dict) -> int:
    """
    Compute the navigator input feature dimension.

    feat = [nu (K + K*D) | dF (1) | topo_feats (n_q + 7 + 4*C_max)]
    Use K_max to ensure feat_dim is large enough for all episode sizes.
    """
    K_max = cfg["problem"]["K_max"]
    D = cfg["problem"]["D"]
    C_max = cfg["navigator"]["C_max"]
    n_q = len(cfg["navigator"]["tau_quantiles"])

    nu_dim = K_max + K_max * D      # sufficient stat residual
    dF_dim = 1                       # energy change
    topo_dim = n_q + 7 + 4 * C_max  # from TopoDiagnostics.feat_dim

    return nu_dim + dF_dim + topo_dim


def build_navigator(cfg: dict, feat_dim: int, phi_dim_max: int) -> GMMNavigator:
    """Instantiate navigator from config."""
    nav_cfg = cfg["navigator"]
    return GMMNavigator(
        feat_dim=feat_dim,
        gru_hidden=nav_cfg["gru_hidden"],
        phi_dim=phi_dim_max,
        port_rank=nav_cfg["port_rank"],
        C_max=nav_cfg["C_max"],
        M_type=nav_cfg["M_type"],
        dt_scale_min=nav_cfg["dt_scale_min"],
        dt_scale_max=nav_cfg["dt_scale_max"],
        gamma_min=nav_cfg["gamma_min"],
        gamma_max=nav_cfg["gamma_max"],
        u_max=nav_cfg["u_max"],
        mlp_hidden=nav_cfg["mlp_hidden"],
        control_mode=cfg["control"]["mode"],
    )


def main():
    parser = argparse.ArgumentParser(description="Train GMM SPHS Navigator")
    parser.add_argument("--config", default="configs/gmm.yaml")
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--K", type=int, default=None, help="Fix K for all episodes")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Overrides from command line
    if args.n_episodes:
        cfg["training"]["n_episodes"] = args.n_episodes
    if args.device:
        cfg["device"] = args.device
    if args.K:
        cfg["problem"]["K_min"] = args.K
        cfg["problem"]["K_max"] = args.K

    device = torch.device(cfg.get("device", "cpu"))
    if cfg.get("seed"):
        torch.manual_seed(cfg["seed"])

    # Compute dimensions
    K_max = cfg["problem"]["K_max"]
    D = cfg["problem"]["D"]
    phi_dim_max = compute_phi_dim(K_max, D)
    feat_dim = compute_feat_dim(cfg)

    print(f"Configuration:")
    print(f"  K range: [{cfg['problem']['K_min']}, {K_max}]")
    print(f"  D: {D}")
    print(f"  phi_dim (K_max): {phi_dim_max}")
    print(f"  feat_dim: {feat_dim}")
    print(f"  n_particles: {cfg['particle']['n_particles']}")
    print(f"  T_batches: {cfg['problem']['T_batches']}")
    print(f"  n_episodes: {cfg['training']['n_episodes']}")
    print(f"  control_mode: {cfg['control']['mode']}")
    print(f"  device: {device}")

    # Build navigator
    navigator = build_navigator(cfg, feat_dim, phi_dim_max).to(device)
    print(f"\nNavigator parameters: {sum(p.numel() for p in navigator.parameters()):,}")

    # Build trainer
    # Add n_particles to problem_cfg for trainer
    problem_cfg = cfg["problem"].copy()
    problem_cfg["n_particles"] = cfg["particle"]["n_particles"]

    trainer = EpisodeTrainer(
        navigator=navigator,
        problem_cfg=problem_cfg,
        integrator_cfg=cfg["integrator"],
        topology_cfg={**cfg["topology"], **cfg["navigator"]},
        training_cfg=cfg["training"],
        device=device,
        output_dir=args.output,
    )

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()

    # Save loss history
    import json
    out_path = Path(args.output) / "loss_history.json"
    with open(out_path, "w") as f:
        json.dump(trainer.loss_history, f, indent=2)
    print(f"Loss history saved to {out_path}")


if __name__ == "__main__":
    main()