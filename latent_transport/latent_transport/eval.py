import csv
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/mnt/data/mplconfig')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DemoConfig
from .metrics import fit_metrics
from .model import LatentPHTransport
from .tasks import make_family_sequences, set_seed, build_online_endpoint_torch


def _stack_step_one(seq, step_idx, key, device):
    arr = seq['steps'][step_idx][key]
    return torch.tensor(arr[None, ...], dtype=torch.float32, device=device)


def _plot_assimilation(records, out_path, title):
    T_steps = len(records)
    fig, axes = plt.subplots(T_steps, 3, figsize=(9.3, 3.0 * T_steps))
    if T_steps == 1:
        axes = np.array([axes])

    gmm = records[0]['gmm']
    xs = np.linspace(-5.0, 5.0, 150)
    ys = np.linspace(-5.0, 5.0, 150)
    Xg, Yg = np.meshgrid(xs, ys)
    grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    Z = np.exp(gmm.log_prob(grid)).reshape(Xg.shape)
    titles = ['Current source', 'Online endpoint guidance', 'Transported cloud']

    for i, rec in enumerate(records):
        metric_txt = (
            f"ED={rec['energy_dist']:.2f}  "
            f"RMSE={rec['center_rmse']:.2f}  "
            f"Khat={int(rec['mode_count_hat'])}"
        )
        for j, key in enumerate(['source', 'target', 'pred']):
            ax = axes[i, j]
            ax.contour(Xg, Yg, Z, levels=6, linewidths=0.8)
            ax.scatter(rec[key][:, 0], rec[key][:, 1], s=12, alpha=0.65)
            ax.scatter(gmm.means[:, 0], gmm.means[:, 1], marker='*', s=140)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(titles[j])
            if j == 0:
                ax.set_ylabel(f"Step {i + 1}\n{metric_txt}")

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate_checkpoint(cfg: DemoConfig, ckpt_path: str):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device(cfg.device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = LatentPHTransport(cfg.hidden, cfg.context_dim, cfg.latent_dim).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    test_family = make_family_sequences(
        num_sequences=cfg.test_sequences,
        K=cfg.K,
        T_steps=cfg.T_steps,
        batch_size_obs=cfg.batch_size_obs,
        M_particles=cfg.M_particles,
    )

    closed_metrics = []
    tf_metrics = []

    representative_closed = None
    representative_tf = None

    with torch.no_grad():
        for si, seq in enumerate(test_family):
            gmm = seq['gmm']

            current = seq['initial_source'].copy()
            seq_rec_closed = []

            current_tf = seq['initial_source'].copy()
            seq_rec_tf = []

            for step_idx in range(cfg.T_steps):
                obs_batch = _stack_step_one(seq, step_idx, 'obs_batch', device)
                step_frac = _stack_step_one(seq, step_idx, 'step_frac', device)
                true_means = _stack_step_one(seq, step_idx, 'true_means', device)
                true_weights = _stack_step_one(seq, step_idx, 'true_weights', device)
                true_stds = _stack_step_one(seq, step_idx, 'true_stds', device)

                src = torch.tensor(current[None, ...], dtype=torch.float32, device=device)
                tgt = build_online_endpoint_torch(
                    src=src,
                    obs_batch=obs_batch,
                    true_means=true_means,
                    true_weights=true_weights,
                    true_stds=true_stds,
                    teacher_mult=cfg.teacher_mult,
                    teacher_jitter=cfg.teacher_jitter,
                )
                pred, _, _, _ = model.rollout_decoded(src, tgt, step_frac, cfg.n_rollout_steps)
                pred_np = pred[0].cpu().numpy()
                tgt_np = tgt[0].cpu().numpy()

                m = fit_metrics(pred_np, gmm, cfg.K)
                m['sequence'] = si
                m['step'] = step_idx + 1
                closed_metrics.append(m)

                seq_rec_closed.append({
                    'source': current.copy(),
                    'target': tgt_np.copy(),
                    'pred': pred_np.copy(),
                    'gmm': gmm,
                    **m,
                })

                current = pred_np.astype(np.float32)

                src_tf = torch.tensor(current_tf[None, ...], dtype=torch.float32, device=device)
                tgt_tf = build_online_endpoint_torch(
                    src=src_tf,
                    obs_batch=obs_batch,
                    true_means=true_means,
                    true_weights=true_weights,
                    true_stds=true_stds,
                    teacher_mult=cfg.teacher_mult,
                    teacher_jitter=cfg.teacher_jitter,
                )
                pred_tf, _, _, _ = model.rollout_decoded(src_tf, tgt_tf, step_frac, cfg.n_rollout_steps)
                pred_tf_np = pred_tf[0].cpu().numpy()
                tgt_tf_np = tgt_tf[0].cpu().numpy()

                m_tf = fit_metrics(pred_tf_np, gmm, cfg.K)
                m_tf['sequence'] = si
                m_tf['step'] = step_idx + 1
                tf_metrics.append(m_tf)

                seq_rec_tf.append({
                    'source': current_tf.copy(),
                    'target': tgt_tf_np.copy(),
                    'pred': pred_tf_np.copy(),
                    'gmm': gmm,
                    **m_tf,
                })

                current_tf = tgt_tf_np.astype(np.float32)

            if representative_closed is None:
                representative_closed = seq_rec_closed
            if representative_tf is None:
                representative_tf = seq_rec_tf

    with open(os.path.join(cfg.out_dir, 'metrics_closed_loop.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(closed_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(closed_metrics)

    with open(os.path.join(cfg.out_dir, 'metrics_teacher_forced.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(tf_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(tf_metrics)

    def summarize(metric_rows):
        summary = {}
        for k in ['center_rmse', 'weight_l1', 'pred_nll', 'oracle_nll', 'energy_dist', 'mode_count_acc']:
            vals = np.array([m[k] for m in metric_rows], dtype=float)
            summary[k] = {'mean': float(vals.mean()), 'std': float(vals.std())}
        summary['mode_count_hat_mean'] = float(np.mean([m['mode_count_hat'] for m in metric_rows]))
        return summary

    summary = {
        'closed_loop': summarize(closed_metrics),
        'teacher_forced': summarize(tf_metrics),
        'history': ckpt.get('history', []),
    }

    with open(os.path.join(cfg.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    hist = ckpt.get('history', [])
    plt.figure(figsize=(5.0, 3.1))
    plt.plot(hist, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.title('Training curve')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, 'training_curve.png'), dpi=180)
    plt.close()

    if representative_closed is not None:
        _plot_assimilation(
            representative_closed,
            os.path.join(cfg.out_dir, 'closed_loop_assimilation.png'),
            'Closed-loop sequential assimilation',
        )

    if representative_tf is not None:
        _plot_assimilation(
            representative_tf,
            os.path.join(cfg.out_dir, 'teacher_forced_assimilation.png'),
            'Teacher-forced one-step assimilation',
        )

    steps = sorted({m['step'] for m in closed_metrics})
    center_closed = [np.mean([m['center_rmse'] for m in closed_metrics if m['step'] == s]) for s in steps]
    center_tf = [np.mean([m['center_rmse'] for m in tf_metrics if m['step'] == s]) for s in steps]

    energy_closed = [np.mean([m['energy_dist'] for m in closed_metrics if m['step'] == s]) for s in steps]
    energy_tf = [np.mean([m['energy_dist'] for m in tf_metrics if m['step'] == s]) for s in steps]

    acc_closed = [np.mean([m['mode_count_acc'] for m in closed_metrics if m['step'] == s]) for s in steps]
    acc_tf = [np.mean([m['mode_count_acc'] for m in tf_metrics if m['step'] == s]) for s in steps]

    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.2))

    ax[0].plot(steps, center_closed, marker='o', label='closed-loop')
    ax[0].plot(steps, center_tf, marker='s', label='teacher-forced')
    ax[0].set_title('Center RMSE')
    ax[0].set_xlabel('Step')
    ax[0].legend()

    ax[1].plot(steps, energy_closed, marker='o', label='closed-loop')
    ax[1].plot(steps, energy_tf, marker='s', label='teacher-forced')
    ax[1].set_title('Energy distance')
    ax[1].set_xlabel('Step')
    ax[1].legend()

    ax[2].plot(steps, acc_closed, marker='o', label='closed-loop')
    ax[2].plot(steps, acc_tf, marker='s', label='teacher-forced')
    ax[2].set_title('Mode-count accuracy')
    ax[2].set_xlabel('Step')
    ax[2].set_ylim(0, 1.05)
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, 'stepwise_metrics.png'), dpi=180)
    plt.close(fig)

    return summary