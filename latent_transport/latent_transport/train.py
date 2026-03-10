import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from .config import DemoConfig
from .model import LatentPHTransport, cubic_bridge
from .ot import sinkhorn_coupling, sinkhorn_distance, sample_ot_targets
from .tasks import make_family_sequences, set_seed, build_online_endpoint_torch


def stack_initial(seq_batch):
    return torch.tensor(
        np.stack([seq['initial_source'] for seq in seq_batch], axis=0),
        dtype=torch.float32,
    )


def stack_step(seq_batch, step_idx, key):
    return torch.tensor(
        np.stack([seq['steps'][step_idx][key] for seq in seq_batch], axis=0),
        dtype=torch.float32,
    )


def cloud_cov(x: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(dim=1, keepdim=True)
    return torch.matmul(xc.transpose(1, 2), xc) / max(x.shape[1], 1)


def soft_basin_loss(
    x_pred: torch.Tensor,
    true_means: torch.Tensor,
    true_weights: torch.Tensor,
    true_stds: torch.Tensor,
    temp: float = 0.35,
) -> torch.Tensor:
    dist2 = torch.cdist(x_pred, true_means, p=2) ** 2
    logits = -dist2 / temp
    a = torch.softmax(logits, dim=-1)

    mass_hat = a.mean(dim=1)
    denom = a.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)

    centers_hat = torch.einsum('bnk,bnd->bkd', a, x_pred) / denom
    diff = x_pred[:, :, None, :] - centers_hat[:, None, :, :]
    var_hat = (a.unsqueeze(-1) * diff.pow(2)).sum(dim=1) / denom

    loss_mass = F.l1_loss(mass_hat, true_weights)
    loss_center = F.mse_loss(centers_hat, true_means)
    loss_var = F.mse_loss(var_hat, true_stds.pow(2))

    return loss_mass + loss_center + 0.5 * loss_var


def teacher_forcing_ratio(cfg: DemoConfig, ep: int) -> float:
    if cfg.epochs <= 1:
        return cfg.teacher_forcing_end
    alpha = ep / (cfg.epochs - 1)
    return (1 - alpha) * cfg.teacher_forcing_start + alpha * cfg.teacher_forcing_end


def train_model(cfg: DemoConfig) -> str:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device(cfg.device)

    train_family = make_family_sequences(
        num_sequences=cfg.train_sequences,
        K=cfg.K,
        T_steps=cfg.T_steps,
        batch_size_obs=cfg.batch_size_obs,
        M_particles=cfg.M_particles,
    )

    model = LatentPHTransport(cfg.hidden, cfg.context_dim, cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []

    for ep in range(cfg.epochs):
        ep_losses = []
        tf_ratio = teacher_forcing_ratio(cfg, ep)

        for _ in range(cfg.steps_per_epoch):
            idx = np.random.choice(len(train_family), size=cfg.batch_tasks, replace=True)
            seq_batch = [train_family[i] for i in idx]

            current = stack_initial(seq_batch).to(device)
            total_loss = 0.0

            for step_idx in range(cfg.T_steps):
                obs_batch = stack_step(seq_batch, step_idx, 'obs_batch').to(device)
                step_frac = stack_step(seq_batch, step_idx, 'step_frac').to(device)
                true_means = stack_step(seq_batch, step_idx, 'true_means').to(device)
                true_weights = stack_step(seq_batch, step_idx, 'true_weights').to(device)
                true_stds = stack_step(seq_batch, step_idx, 'true_stds').to(device)

                tgt = build_online_endpoint_torch(
                    src=current,
                    obs_batch=obs_batch,
                    true_means=true_means,
                    true_weights=true_weights,
                    true_stds=true_stds,
                    teacher_mult=cfg.teacher_mult,
                    teacher_jitter=cfg.teacher_jitter,
                )

                c = model.context(current, tgt, step_frac)
                z_src = model.encode(current, c)
                z_tgt = model.encode(tgt, c).detach()

                # local latent bridge supervision using sampled OT matches, not row barycenters
                P_src = sinkhorn_coupling(current.detach(), tgt.detach(), eps=cfg.sinkhorn_eps, iters=cfg.sinkhorn_iters)
                z_match = sample_ot_targets(P_src.detach(), z_tgt.detach())

                tau = torch.rand(current.shape[0], current.shape[1], 1, device=device)
                z_tau, r_tau, dr_tau = cubic_bridge(z_src, z_match, tau)
                c_rep = c[:, None, :].expand(-1, current.shape[1], -1)
                _, dr_pred, u = model.field(z_tau, r_tau, tau, c_rep)
                loss_match = F.mse_loss(dr_pred, dr_tau)

                x_pred, zT, _, act = model.rollout_decoded(current, tgt, step_frac, cfg.n_rollout_steps)

                # terminal losses are setwise, not MSE-to-barycenter
                loss_endpoint = sinkhorn_distance(
                    x_pred, tgt,
                    eps=cfg.sinkhorn_eps,
                    iters=cfg.sinkhorn_iters,
                )
                loss_lat_end = sinkhorn_distance(
                    zT, z_tgt,
                    eps=cfg.sinkhorn_eps,
                    iters=cfg.sinkhorn_iters,
                )
                loss_cov = F.mse_loss(cloud_cov(x_pred), cloud_cov(tgt))
                loss_basin = soft_basin_loss(x_pred, true_means, true_weights, true_stds)

                step_loss = (
                    cfg.w_match * loss_match
                    + cfg.w_lat_end * loss_lat_end
                    + cfg.w_endpoint * loss_endpoint
                    + cfg.w_cov * loss_cov
                    + cfg.w_basin * loss_basin
                    + cfg.w_control * act
                )

                total_loss = total_loss + step_loss

                if np.random.rand() < tf_ratio:
                    current = tgt.detach()
                else:
                    current = x_pred.detach()

            total_loss = total_loss / cfg.T_steps

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            ep_losses.append(float(total_loss.item()))

        history.append(float(np.mean(ep_losses)))
        print(f'Epoch {ep + 1:02d} loss={history[-1]:.4f} tf_ratio={tf_ratio:.2f}')

    ckpt_path = os.path.join(cfg.out_dir, 'checkpoint.pt')
    torch.save(
        {
            'state_dict': model.state_dict(),
            'config': cfg.__dict__,
            'history': history,
        },
        ckpt_path,
    )

    with open(os.path.join(cfg.out_dir, 'train_history.json'), 'w') as f:
        json.dump({'history': history}, f, indent=2)

    return ckpt_path