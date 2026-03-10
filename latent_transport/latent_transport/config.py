from dataclasses import dataclass

@dataclass
class DemoConfig:
    seed: int = 42
    device: str = 'cuda'
    out_dir: str = './results_demo'

    K: int = 3
    T_steps: int = 8
    batch_size_obs: int = 20
    train_sequences: int = 12
    test_sequences: int = 4
    M_particles: int = 24

    hidden: int = 32
    context_dim: int = 12
    latent_dim: int = 2

    epochs: int = 50
    steps_per_epoch: int = 8
    batch_tasks: int = 3
    lr: float = 2e-3

    n_rollout_steps: int = 12

    sinkhorn_eps: float = 0.12
    sinkhorn_iters: int = 30
    teacher_mult: int = 3
    teacher_jitter: float = 0.02

    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.0

    w_match: float = 0.20
    w_lat_end: float = 0.50
    w_endpoint: float = 1.50
    w_cov: float = 1.00
    w_basin: float = 0.75
    w_control: float = 1e-4