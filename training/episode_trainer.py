"""
Episode Trainer.

Implements the three-loop training structure:
    Outer: episodes (sample new GMM problem)
    Middle: streaming batches (sequential Bayesian update)
    Inner: BAOAB steps (within-batch transport)

The navigator is trained end-to-end through the unrolled middle loop.
PHC topology features are stop-gradded (non-differentiable ops).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from data.episode_sampler import EpisodeSampler
from data.gmm_problem import GMMProblem, phi_dim as compute_phi_dim
from dynamics.gmm_energy import GMMEnergy
from dynamics.baoab import BAOABIntegrator, smc_weight_update, compute_ess, systematic_resample
from dynamics.canonicalize import canonicalize_phi
from fidelity.casimir import CasimirChecker
from models.gmm_embedder import GMMEmbedder
from models.gmm_navigator import GMMNavigator, build_features, TransportParams, compute_per_particle_Gu
from topology.phc import PHC
from topology.diagnostics import TopoDiagnostics
from training.losses import combined_episode_loss


class EpisodeTrainer:
    """
    Trains the GMMNavigator via episodic unrolled learning.

    Training loop:
        for episode in n_episodes:
            sample problem (K, D, overlap)
            initialize phi ~ prior
            for k in T_batches:
                sense(phi, X_k) → nu, dF
                PHC(phi, weights) → topo (stop-grad)
                navigator(feat) → Theta_k     [DIFFERENTIABLE]
                BAOAB(phi, Theta_k, X_k)      [DIFFERENTIABLE]
                SMC weight update             [STOP-GRAD]
                maybe resample
            compute_loss(phi_final, problem) → backward
            optimizer.step()
    """

    def __init__(
        self,
        # Navigator (the only learned component)
        navigator: GMMNavigator,
        # Config dicts
        problem_cfg: dict,
        integrator_cfg: dict,
        topology_cfg: dict,
        training_cfg: dict,
        # Device
        device: torch.device = torch.device("cpu"),
        # Output
        output_dir: str = "outputs/",
    ):
        self.navigator = navigator
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Unpack configs
        self.K_min         = problem_cfg["K_min"]
        self.K_max         = problem_cfg["K_max"]
        self.D             = problem_cfg["D"]
        self.N             = problem_cfg["N"]
        self.T_batches     = problem_cfg["T_batches"]
        self.batch_size    = problem_cfg["batch_size"]
        self.n_particles   = problem_cfg.get("n_particles", 64)

        self.h             = integrator_cfg["h"]
        self.K_steps       = integrator_cfg["K_steps"]
        self.default_friction = integrator_cfg["friction"]
        self.default_mass     = integrator_cfg["mass"]

        self.sigma_rule    = topology_cfg["sigma_rule"]
        self.C_max         = topology_cfg.get("C_max", 8)
        self.tau_quantiles = topology_cfg.get("tau_quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
        self.ESS_min       = topology_cfg.get("ESS_min", 5.0)
        self.resample_thr  = topology_cfg.get("resample_threshold", 0.5)
        self.knn_k         = topology_cfg.get("knn_k", 3)
        self.tau_op_q      = topology_cfg.get("tau_operational_quantile", 0.1)

        self.n_episodes    = training_cfg["n_episodes"]
        self.lr            = training_cfg["lr"]
        self.grad_clip     = training_cfg.get("grad_clip", 1.0)
        self.log_interval  = training_cfg.get("log_interval", 50)
        self.ckpt_interval = training_cfg.get("checkpoint_interval", 500)
        self.lambda_terminal = training_cfg.get("lambda_terminal", 1.0)
        self.lambda_topo     = training_cfg.get("lambda_topo", 0.1)
        self.lambda_ess      = training_cfg.get("lambda_ess", 0.1)
        self.lambda_casimir  = training_cfg.get("lambda_casimir", 0.05)
        # Gradient accumulation: average gradients over N episodes before stepping.
        # Reduces gradient variance by sqrt(n_accumulate).
        self.n_accumulate    = training_cfg.get("n_accumulate", 4)
        # Partial unroll: keep last n_grad_batches in the gradient graph.
        # Each kept batch contributes an intermediate loss.  This gives direct
        # gradient signal to those batches' navigator outputs (not just via
        # h_gru BPTT), teaching the GRU *when* to change beta/M, not just *how*.
        self.n_grad_batches  = training_cfg.get("n_grad_batches", 3)
        # Replay buffer: BAOAB force uses last n batches concatenated.
        # n_replay=3, cap=150 means ~3 batches of 50 = 150 points max — negligible
        # compute overhead vs a single batch, but force approximates 3x more
        # of the accumulated posterior than X_k alone.
        # replay_buffer_size no longer needed (replaced by Vitter reservoir sampling)
        self.replay_cap_points  = training_cfg.get("replay_cap_points", 150)

        self.control_mode  = navigator.control_mode

        # Optimizer + OneCycleLR schedule
        # OneCycleLR: single triangle (linear warmup → cosine anneal → constant).
        # No cycling — LR stays at min_lr after the peak, preventing the
        # late-training explosions we saw with CosineAnnealingLR.
        #
        # n_accumulate episodes per optimizer step → total_steps = ceil(n_episodes / n_accumulate)
        n_accumulate = training_cfg.get("n_accumulate", 4)
        total_opt_steps = max(1, math.ceil(self.n_episodes / n_accumulate))
        pct_start = 0.15   # spend 15% of steps warming up (typical for meta-learning)

        self.optimizer = Adam(navigator.parameters(), lr=self.lr)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=total_opt_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=10.0,        # start at lr/10
            final_div_factor=100.0, # end at lr/1000, never cycles back
        )

        # Episode sampler
        self.sampler = EpisodeSampler(
            K_min=self.K_min,
            K_max=self.K_max,
            D=self.D,
            N=self.N,
            T_batches=self.T_batches,
            batch_size=self.batch_size,
            n_particles=self.n_particles,
            device=device,
        )

        # Training history
        self.loss_history: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Core episode runner
    # ─────────────────────────────────────────────────────────────────────────

    def run_episode(
        self,
        episode: Dict,
        train: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[TopoDiagnostics], Dict]:
        """
        Run a single episode (middle + inner loops).

        Args:
            episode: dict from EpisodeSampler.sample_episode()
            train:   if False, stop all gradients (eval mode)
        Returns:
            phi_final:    [M, phi_dim] terminal particles
            weights_final:[M] terminal weights
            topo_history: list of T TopoDiagnostics
            info:         diagnostic dict
        """
        problem: GMMProblem = episode["problem"]
        batches: List[torch.Tensor] = episode["batches"]
        phi_init: torch.Tensor = episode["phi_init"]
        K: int = episode["K"]

        D = self.D
        pd = compute_phi_dim(K, D)

        # Build per-episode components
        energy_fn = GMMEnergy(K=K, D=D).to(self.device)
        embedder  = GMMEmbedder(K=K, D=D).to(self.device)
        phc = PHC(
            embedder=embedder,
            K=K, D=D,
            sigma_rule=self.sigma_rule,
            C_max=self.C_max,
            tau_quantiles=self.tau_quantiles,
            ESS_min=self.ESS_min,
            knn_k=self.knn_k,
            tau_operational_quantile=self.tau_op_q,
        )
        integrator = BAOABIntegrator(phi_dim=pd, friction=self.default_friction, dt=self.h)
        casimir = CasimirChecker(K=K, D=D)

        # Initialize
        phi = phi_init.clone().to(self.device)              # [M, pd]
        log_weights = torch.zeros(phi.shape[0], device=self.device)
        weights = torch.ones(phi.shape[0], device=self.device) / phi.shape[0]

        topo_history: List[TopoDiagnostics] = []
        X_prev: Optional[torch.Tensor] = None
        h_gru: Optional[torch.Tensor] = None
        F_prev: Optional[float] = None

        info = {"ess_curve": [], "C_tau_curve": [], "energy_curve": []}

        ctx = torch.enable_grad() if train else torch.no_grad()

        n_batches = len(batches)
        # n_grad_batches: keep the last N batches' phi in the autograd graph.
        # These batches each contribute an intermediate loss, giving the navigator
        # direct per-batch gradient signal rather than only via h_gru BPTT.
        n_grad = self.n_grad_batches if train else 0
        grad_start = max(0, n_batches - n_grad)   # batches [grad_start, T) are kept

        phi_snapshots = []   # (phi_in_graph, X_k, weights) for intermediate losses

        # Reservoir for BAOAB force (Vitter's Algorithm R).
        # Maintains a uniform random sample of ALL data seen so far (up to
        # replay_cap points) so that BAOAB force approximates the gradient of
        # the FULL accumulated log-likelihood, not just the last batch.
        #
        # Why reservoir sampling instead of a sliding window:
        #   Sliding window of R batches → BAOAB targets p(phi | X_{k-R:k})
        #   Reservoir of all data seen → BAOAB targets p(phi | X_{1:k})  [unbiased]
        # The reservoir makes BAOAB a consistent rejuvenation kernel for the
        # SMC posterior weights, which also represent p(phi | X_{1:k}).
        #
        # SMC weights still use beta=1 (true posterior); BAOAB beta remains a
        # navigator-tuned PROPOSAL parameter that doesn't change the target.
        replay_cap  = self.replay_cap_points   # max reservoir size
        res_buf     = None   # [replay_cap, D] reservoir tensor (allocated lazily)
        n_seen      = 0      # total data points processed so far this episode
        res_fill    = 0      # how many reservoir slots are occupied

        with ctx:
            for k, X_k in enumerate(batches):
                X_k = X_k.to(self.device)
                M = phi.shape[0]
                is_last_batch = (k == n_batches - 1)
                in_grad_window = train and (k >= grad_start)

                assert phi.shape == (M, pd), \
                    f"[batch {k}] phi shape {phi.shape} != expected ({M}, {pd})"

                # ── Update reservoir (Vitter Algorithm R) ─────────────────────
                Xd = X_k.detach()         # [B, D]
                B_k, D_x = Xd.shape
                if res_buf is None:
                    res_buf = torch.empty(replay_cap, D_x, device=self.device)
                for i in range(B_k):
                    idx_global = n_seen + i
                    if res_fill < replay_cap:
                        res_buf[res_fill] = Xd[i]
                        res_fill += 1
                    else:
                        j = int(torch.randint(0, idx_global + 1, (1,)).item())
                        if j < replay_cap:
                            res_buf[j] = Xd[i]
                n_seen += B_k
                X_replay = res_buf[:res_fill]   # [res_fill, D] — filled portion only

                # ── Sense ────────────────────────────────────────────────────
                phi_mean = (weights.unsqueeze(-1) * phi).sum(0).detach()
                nu, dF = energy_fn.sense(phi_mean, X_k, X_prev)

                # ── PHC (stop-grad) ──────────────────────────────────────────
                with torch.no_grad():
                    topo, filtration = phc.run(
                        phi.detach(), weights.detach(), X_ref=X_k.detach()
                    )
                topo_history.append(topo)

                # ── Build features ───────────────────────────────────────────
                feat = build_features(
                    nu=nu.detach(),
                    dF=dF.detach(),
                    topo=topo,
                    tau_quantiles=self.tau_quantiles,
                    target_feat_dim=self.navigator.feat_dim,
                ).to(self.device)

                assert feat.shape == (self.navigator.feat_dim,), \
                    f"[batch {k}] feat shape {feat.shape} != ({self.navigator.feat_dim},)"

                # ── Navigate ─────────────────────────────────────────────────
                # h_gru is never detached — full BPTT through hidden state.
                params, h_gru = self.navigator.forward_step(feat, h_gru)
                params = self.navigator.topology_adjustment(
                    params, topo, C_target=K, ESS_min=self.ESS_min
                )

                # ── Navigator outputs — keep as tensors ───────────────────────
                M_diag_ep  = params.M_diag[:pd]
                gamma_ep   = params.gamma
                dt_scale_ep = params.dt_scale   # target-neutral timestep multiplier
                accumulate_girsanov = (self.control_mode == "B")

                assert M_diag_ep.shape == (pd,), \
                    f"[batch {k}] M_diag_ep shape {M_diag_ep.shape} != ({pd},)"

                # ── Standard SMC-sampler order: reweight → resample → rejuvenate
                #
                # Reweight first (using new X_k) so weights reflect p(phi|X_{1:k})
                # BEFORE we move particles. Resampling then focuses compute on
                # high-weight regions. BAOAB then rejuvenates using the SAME target
                # the weights represent (via the reservoir). This eliminates the
                # "proposal depends on X_k but weights don't account for it" mismatch.

                # ── 1) Reweight with X_k ──────────────────────────────────────
                with torch.no_grad():
                    log_weights, weights = smc_weight_update(
                        log_weights=log_weights,
                        phi=phi.detach(),
                        X_k=X_k,
                        energy_fn=energy_fn,
                        beta=1.0,   # always 1 — weights track true posterior
                        girsanov_log_w=None,   # Girsanov correction added after BAOAB
                    )
                    ess = compute_ess(weights)

                # ── 2) Resample if ESS too low (skip inside grad window) ───────
                with torch.no_grad():
                    if ess < self.resample_thr * M and not in_grad_window:
                        phi_r, weights = systematic_resample(phi.detach(), weights, self.device)
                        phi = phi_r.requires_grad_(True) if train else phi_r
                        log_weights = torch.log(weights.clamp(min=1e-30))

                # ── 3) Compute per-particle Gu for mode B ─────────────────────
                # G [phi_dim, port_rank] projects into the navigator's learned
                # subspace; u [C_max] gives per-cluster amplitudes.
                # compute_per_particle_Gu maps these to Gu [M, phi_dim] where
                # each particle is steered toward the minimum-mass cluster centroid
                # with amplitude u[c_m] — heavy-cluster particles steer hard,
                # sparse-cluster particles are left mostly undisturbed.
                if self.control_mode == "B":
                    G_ep = params.G[:pd, :]   # [phi_dim, port_rank]
                    u_ep = params.u           # [C_max]
                    Gu_ep = compute_per_particle_Gu(
                        phi=phi.detach(),
                        assignments=filtration.assignments,
                        G=G_ep,
                        u_per_cluster=u_ep,
                        W_c=topo.W_c,
                    )                         # [M, phi_dim]
                else:
                    Gu_ep = None

                # ── 4) Rejuvenate with BAOAB (targets same posterior as weights)
                # Force uses X_replay (reservoir ≈ all data seen so far) so BAOAB
                # targets p(phi|X_{1:k}), matching the weight target exactly.
                grad_fn = energy_fn.make_grad_fn(
                    X_replay, create_graph=in_grad_window
                )
                result = integrator.integrate(
                    phi0=phi,
                    grad_fn=grad_fn,
                    n_steps=self.K_steps,
                    friction=gamma_ep,
                    M_diag=M_diag_ep,
                    Gu=Gu_ep,
                    accumulate_girsanov=accumulate_girsanov,
                    dt_scale=dt_scale_ep,
                )
                phi = result["phi"]
                girsanov_log_w = result["girsanov_log_w"]

                # ── 5) Girsanov correction for mode B (after rejuvenation) ─────
                if accumulate_girsanov:
                    with torch.no_grad():
                        log_weights = log_weights + girsanov_log_w.detach()
                        log_weights = log_weights - log_weights.max()
                        weights = torch.softmax(log_weights, dim=0)

                # ── Canonicalize between batches ──────────────────────────────
                # Done BEFORE snapshot so the loss always sees canonical phi.
                # Inside grad window: straight-through estimator keeps the
                #   transport chain connected across batches while presenting
                #   permutation-stable phi to the loss.
                # Outside grad window: hard detach (saves memory).
                if in_grad_window and train:
                    phi_c = canonicalize_phi(phi.detach(), K, D)
                    # Straight-through: forward value is canonical; backward
                    # gradient passes through as identity (no argsort in grad path)
                    phi = phi + (phi_c - phi).detach()
                else:
                    phi = canonicalize_phi(phi.detach(), K, D)
                    if not is_last_batch:
                        phi = phi.requires_grad_(True) if train else phi

                # ── Snapshot canonical phi for intermediate/terminal loss ─────
                # Taken AFTER canonicalization so loss sees canonical phi.
                # Weights are post-update (consistent (phi, w) pair) and
                # tempered (clip at 2/M) to cap gradient variance when ESS tiny.
                if in_grad_window:
                    recency_w = 0.5 ** (n_batches - 1 - k)
                    w_tempered = weights.detach().clamp(max=2.0 / M)
                    w_tempered = w_tempered / w_tempered.sum()
                    phi_snapshots.append((phi, X_k.detach(), w_tempered, recency_w))

                # ── Diagnostics ──────────────────────────────────────────────
                with torch.no_grad():
                    F_curr = energy_fn.free_energy(phi.detach(), X_k).mean().item()

                info["ess_curve"].append(ess)
                info["C_tau_curve"].append(topo.C_tau)
                info["energy_curve"].append(F_curr)

                X_prev = X_k
                F_prev = F_curr

        # Return: list of snapshots for multi-step loss, plus final phi for inference
        return phi_snapshots, phi, weights, topo_history, info

    # ─────────────────────────────────────────────────────────────────────────
    # Training outer loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        """
        Main training loop with gradient accumulation + partial unroll losses.

        run_episode returns:
            phi_snapshots  — list of (phi, X_k, weights, recency_w) for last
                             n_grad_batches, each still in the autograd graph
            phi_inf        — final phi (detached), for diagnostics only
            weights        — final SMC weights
            topo_history   — list of TopoDiagnostics
            info           — dict with ess_curve, C_tau_curve, energy_curve

        Loss per episode = sum_k recency_w_k * terminal_loss(phi_snapshot_k)
        This gives direct gradient to each kept batch's navigator outputs,
        not just via h_gru BPTT from the last batch.

        Loss normalisation: divide each episode's loss by its initial mse_mu
        (at batch 0 snapshot), so hard/easy problems contribute equal gradient
        magnitude rather than hard problems dominating.
        """
        print(f"Starting training: {self.n_episodes} episodes")
        print(f"Device: {self.device}")
        print(f"Navigator params: {sum(p.numel() for p in self.navigator.parameters()):,}")
        print(f"Grad accumulation: {self.n_accumulate} episodes/step, "
              f"partial unroll last {self.n_grad_batches} batches")

        best_loss = float("inf")
        t0 = time.time()
        grad_norm = 0.0

        ema_loss  = None
        ema_alpha = 0.1

        self.optimizer.zero_grad()
        accum_count = 0

        for ep in range(self.n_episodes):
            self.navigator.train()
            episode = self.sampler.sample_curriculum_episode(ep, self.n_episodes)

            try:
                # ── Run episode (partial unroll) ──────────────────────────────
                phi_snapshots, phi_inf, weights, topo_history, info = self.run_episode(
                    episode, train=True
                )

                energy_fn_ep = GMMEnergy(K=episode["K"], D=self.D).to(self.device)

                # ── Compute per-snapshot losses, weighted by recency ───────────
                # Anchor normalisation on the first snapshot's MSE so that
                # hard GMM problems (large inter-centroid distance) don't dominate
                # the gradient over easy ones.  Clamp to avoid division by ~0.
                total_loss = torch.tensor(0.0, device=self.device)
                first_mse  = None

                for phi_snap, X_snap, w_snap, rec_w in phi_snapshots:
                    snap_losses = combined_episode_loss(
                        phi_final=phi_snap,
                        weights=w_snap,
                        topo_history=topo_history,
                        casimir_result=None,
                        problem=episode["problem"],
                        energy_fn=energy_fn_ep,
                        lambda_terminal=self.lambda_terminal,
                        lambda_topo=self.lambda_topo,     # NOT scaled by rec_w here
                        lambda_ess=self.lambda_ess,       # rec_w applied once below
                        lambda_casimir=self.lambda_casimir,
                    )
                    snap_mse = snap_losses.get("mse_mu", torch.tensor(1.0))
                    if first_mse is None:
                        first_mse = snap_mse.detach().clamp(min=0.1)
                    # rec_w applied ONCE here — not inside the loss function
                    total_loss = total_loss + rec_w * snap_losses["total"]

                # Normalise by initial difficulty so all episodes contribute
                # equal gradient magnitude independent of problem scale
                norm_loss = total_loss / first_mse.clamp(min=0.1)

                if torch.isfinite(norm_loss):
                    (norm_loss / self.n_accumulate).backward()

                    # ── Per-episode gradient guard ─────────────────────────
                    # Check gradient norm of THIS episode's contribution before
                    # committing it to the accumulation buffer.  A single bad
                    # episode (large curvature / unlucky init) can produce a
                    # gradient 100× larger than typical; even after dividing by
                    # n_accumulate it dominates the accumulated direction.
                    # Threshold: 10× grad_clip.  If exceeded, zero the
                    # contribution now (before it contaminates other episodes).
                    ep_gnorm = torch.nn.utils.clip_grad_norm_(
                        self.navigator.parameters(),
                        self.grad_clip * 10.0,  # per-episode ceiling = 10× step clip
                    ).item()
                    if ep_gnorm > self.grad_clip * 10.0:
                        # Gradient already clipped in-place by clip_grad_norm_.
                        # Still count it (clipped contribution is valid), but
                        # warn so we can diagnose persistent explosion.
                        pass   # clip_grad_norm_ already applied the ceiling
                    accum_count += 1

                losses = {"mse_mu": phi_snapshots[-1][0].detach().mean() if phi_snapshots
                          else torch.tensor(float("nan"))}
                # Re-compute mse_mu for logging from last snapshot
                with torch.no_grad():
                    last_losses = combined_episode_loss(
                        phi_final=phi_snapshots[-1][0].detach(),
                        weights=phi_snapshots[-1][2],
                        topo_history=topo_history,
                        casimir_result=None,
                        problem=episode["problem"],
                        energy_fn=energy_fn_ep,
                        lambda_terminal=1.0,
                        lambda_topo=0.0,
                        lambda_ess=0.0,
                        lambda_casimir=0.0,
                    )
                losses = last_losses
                total_loss_log = total_loss

            except Exception as e:
                import traceback
                print(f"  Episode {ep} failed: {e}")
                traceback.print_exc()
                total_loss_log = torch.tensor(float("nan"))
                losses = {"mse_mu": torch.tensor(float("nan"))}
                info = {"C_tau_curve": [0], "ess_curve": [0.0]}

            # ── Optimizer step every n_accumulate episodes ─────────────────────
            # optimizer.step() then scheduler.step() — correct PyTorch order.
            # OneCycleLR steps once per optimizer step (not per episode),
            # so scheduler.step() is inside this block.
            step_skipped = False
            if (ep + 1) % self.n_accumulate == 0 and accum_count > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.navigator.parameters(), self.grad_clip
                ).item()
                # Skip step if accumulated norm is still too large after per-episode
                # clipping.  Threshold = 20× grad_clip (was 1e4 — far too permissive).
                # At 20× the signal is mostly noise; better to discard and re-accumulate.
                skip_threshold = self.grad_clip * 20.0
                if grad_norm < skip_threshold:
                    self.optimizer.step()
                    self.scheduler.step()
                else:
                    step_skipped = True
                self.optimizer.zero_grad()
                accum_count = 0

            # ── Logging ───────────────────────────────────────────────────────
            loss_val = total_loss_log.item() if torch.is_tensor(total_loss_log) else float(total_loss_log)
            mse_val  = losses.get("mse_mu", torch.tensor(0.0))
            mse_val  = mse_val.item() if torch.is_tensor(mse_val) else float(mse_val)

            if math.isfinite(loss_val):
                ema_loss = loss_val if ema_loss is None else (
                    ema_alpha * loss_val + (1 - ema_alpha) * ema_loss
                )

            record = {
                "episode": ep,
                "loss": loss_val,
                "ema_loss": ema_loss,
                "mse_mu": mse_val,
                "C_tau_final": info["C_tau_curve"][-1],
                "ess_final": info["ess_curve"][-1],
                "K": episode["K"],
                "difficulty": episode.get("difficulty", 0.0),
                "grad_norm": grad_norm,
            }
            self.loss_history.append(record)

            if (ep + 1) % self.log_interval == 0:
                elapsed = time.time() - t0
                # Current LR for monitoring
                cur_lr = self.optimizer.param_groups[0]["lr"]
                # Detect stall: EMA not improving
                if len(self.loss_history) >= 2 * self.log_interval:
                    recent = [r["ema_loss"] for r in self.loss_history[-self.log_interval:]
                              if r.get("ema_loss") is not None]
                    old_ema = [r["ema_loss"] for r in self.loss_history[-2*self.log_interval:-self.log_interval]
                               if r.get("ema_loss") is not None]
                    if recent and old_ema:
                        rel_improve = (sum(old_ema)/len(old_ema) - sum(recent)/len(recent)) / (abs(sum(old_ema)/len(old_ema)) + 1e-8)
                        stall_flag = " [STALL?]" if rel_improve < 0.005 else ""
                    else:
                        stall_flag = ""
                else:
                    stall_flag = ""
                ema_disp = ema_loss if (ema_loss is not None and math.isfinite(ema_loss)) else float("nan")

                print(
                    f"Ep {ep+1:5d}/{self.n_episodes} | "
                    f"loss={loss_val:.4f} (ema={ema_disp:.4f}){stall_flag} | "
                    f"mse_mu={mse_val:.4f} | "
                    f"K={episode['K']} ESS={record['ess_final']:.1f} | "
                    f"gnorm={grad_norm:.3f} lr={cur_lr:.2e} | "
                    f"t={elapsed:.0f}s"
                )

            # Best model: use EMA loss to avoid saving on a lucky episode
            is_best_step = (ep + 1) % self.n_accumulate == 0 and ema_loss is not None
            if is_best_step and ema_loss < best_loss:
                best_loss = ema_loss
                self._save_checkpoint("best", ema_loss)

            if (ep + 1) % self.ckpt_interval == 0:
                self._save_checkpoint(ep + 1, loss_val)

        print(f"\nTraining complete. Best EMA loss: {best_loss:.4f}")
        self._save_checkpoint("final", loss_val)

    # ─────────────────────────────────────────────────────────────────────────
    # Inference (no training)
    # ─────────────────────────────────────────────────────────────────────────

    def infer(
        self,
        problem: GMMProblem,
        batches: List[torch.Tensor],
        phi_init: torch.Tensor,
    ) -> Dict:
        """
        Run inference on a given problem (no gradient computation).

        Returns dict with phi_final, weights, topo_history, info.
        """
        self.navigator.eval()
        episode = {
            "problem": problem,
            "batches": batches,
            "phi_init": phi_init,
            "K": problem.K,
        }
        with torch.no_grad():
            _, phi, weights, topo_history, info = self.run_episode(episode, train=False)

        return {
            "phi": phi,
            "weights": weights,
            "topo_history": topo_history,
            "info": info,
        }

    def infer_with_trajectory(
        self,
        problem: GMMProblem,
        batches: List[torch.Tensor],
        phi_init: torch.Tensor,
    ) -> Dict:
        """
        Run inference and log the FULL trajectory for visualization.

        Returns all per-batch states:
            phi_history:        [T] list of [M, phi_dim]  (after transport)
            weights_history:    [T] list of [M]
            filtration_history: [T] list of FiltrationResult
            topo_history:       [T] list of TopoDiagnostics
            params_history:     [T] list of dicts {dt_scale, gamma, M_diag_norm}
            batches:            [T] list of [B, D] data batches
            info:               {ess_curve, C_tau_curve, energy_curve}
            phi_init:           [M, phi_dim] initial particles
        """
        self.navigator.eval()
        K = problem.K
        D = self.D
        pd = compute_phi_dim(K, D)

        energy_fn = GMMEnergy(K=K, D=D).to(self.device)
        embedder  = GMMEmbedder(K=K, D=D).to(self.device)
        phc = PHC(
            embedder=embedder, K=K, D=D,
            sigma_rule=self.sigma_rule, C_max=self.C_max,
            tau_quantiles=self.tau_quantiles, ESS_min=self.ESS_min,
            knn_k=self.knn_k, tau_operational_quantile=self.tau_op_q,
        )
        integrator = BAOABIntegrator(phi_dim=pd, friction=self.default_friction, dt=self.h)

        phi         = phi_init.clone().to(self.device)
        log_weights = torch.zeros(phi.shape[0], device=self.device)
        weights     = torch.ones(phi.shape[0], device=self.device) / phi.shape[0]

        # Trajectory storage
        phi_history        = [phi_init.clone().cpu()]   # include t=0
        weights_history    = [weights.clone().cpu()]
        filtration_history = []
        topo_history_out   = []
        params_history     = []
        batches_out        = []
        X_prev = None
        h_gru  = None

        info = {"ess_curve": [], "C_tau_curve": [], "energy_curve": []}

        # Vitter reservoir — identical to run_episode so plots show the true method
        replay_cap = self.replay_cap_points
        res_buf    = None
        n_seen     = 0
        res_fill   = 0

        with torch.no_grad():
            for k, X_k in enumerate(batches):
                X_k = X_k.to(self.device)
                M   = phi.shape[0]
                batches_out.append(X_k.cpu())

                # ── Reservoir update (Vitter Algorithm R) ─────────────────────
                Xd = X_k
                B_k, D_x = Xd.shape
                if res_buf is None:
                    res_buf = torch.empty(replay_cap, D_x, device=self.device)
                for i in range(B_k):
                    idx_global = n_seen + i
                    if res_fill < replay_cap:
                        res_buf[res_fill] = Xd[i]
                        res_fill += 1
                    else:
                        j = int(torch.randint(0, idx_global + 1, (1,)).item())
                        if j < replay_cap:
                            res_buf[j] = Xd[i]
                n_seen  += B_k
                X_force  = res_buf[:res_fill]   # accumulated history for BAOAB

                phi_mean = (weights.unsqueeze(-1) * phi).sum(0)
                nu, dF   = energy_fn.sense(phi_mean, X_k, X_prev)

                topo, filtration = phc.run(phi, weights, X_ref=X_k)
                topo_history_out.append(topo)
                filtration_history.append(filtration)

                feat = build_features(
                    nu=nu, dF=dF, topo=topo,
                    tau_quantiles=self.tau_quantiles,
                    target_feat_dim=self.navigator.feat_dim,
                ).to(self.device)

                params, h_gru = self.navigator.forward_step(feat, h_gru)
                params = self.navigator.topology_adjustment(
                    params, topo, C_target=K, ESS_min=self.ESS_min
                )

                dt_scale_val = params.dt_scale.item() if torch.is_tensor(params.dt_scale) else float(params.dt_scale)
                gamma_val    = params.gamma.item()    if torch.is_tensor(params.gamma)    else float(params.gamma)

                params_history.append({
                    "dt_scale":     dt_scale_val,
                    "gamma":        gamma_val,
                    "M_diag_mean":  params.M_diag[:pd].mean().item(),
                    "M_diag_std":   params.M_diag[:pd].std().item(),
                    "batch":        k,
                })

                M_diag_ep = params.M_diag[:pd]
                accumulate_girsanov = (self.control_mode == "B")

                # SMC order: reweight → resample → BAOAB  (matches run_episode)
                log_weights, weights = smc_weight_update(
                    log_weights=log_weights, phi=phi, X_k=X_k,
                    energy_fn=energy_fn, beta=1.0,
                )
                ess = compute_ess(weights)
                if ess < self.resample_thr * M:
                    phi, weights = systematic_resample(phi, weights, self.device)
                    log_weights  = torch.log(weights.clamp(min=1e-30))

                # Per-particle Gu for mode B (same logic as run_episode)
                if self.control_mode == "B":
                    Gu_ep = compute_per_particle_Gu(
                        phi=phi,
                        assignments=filtration.assignments,
                        G=params.G[:pd, :],
                        u_per_cluster=params.u,
                        W_c=topo.W_c,
                    )
                else:
                    Gu_ep = None

                # BAOAB uses X_force (reservoir) to match training-time target
                grad_fn = energy_fn.make_grad_fn(X_force)
                result  = integrator.integrate(
                    phi0=phi, grad_fn=grad_fn, n_steps=self.K_steps,
                    friction=gamma_val, M_diag=M_diag_ep, dt_scale=dt_scale_val,
                    Gu=Gu_ep, accumulate_girsanov=accumulate_girsanov,
                )
                phi = canonicalize_phi(result["phi"], K, D)

                # Mode B: apply Girsanov correction to weights after transport
                if accumulate_girsanov:
                    girsanov_log_w = result["girsanov_log_w"]
                    log_weights = log_weights + girsanov_log_w
                    log_weights = log_weights - log_weights.max()
                    weights     = torch.softmax(log_weights, dim=0)

                # Store post-transport state
                phi_history.append(phi.clone().cpu())
                weights_history.append(weights.clone().cpu())

                F_curr = energy_fn.free_energy(phi, X_force).mean().item()
                info["ess_curve"].append(ess)
                info["C_tau_curve"].append(topo.C_tau)
                info["energy_curve"].append(F_curr)

                X_prev = X_k

        return {
            "phi":               phi.cpu(),
            "weights":           weights.cpu(),
            "phi_history":       phi_history,       # T+1 entries (including t=0)
            "weights_history":   weights_history,   # T+1 entries
            "filtration_history":filtration_history, # T entries
            "topo_history":      topo_history_out,  # T entries
            "params_history":    params_history,    # T entries
            "batches":           batches_out,        # T entries
            "info":              info,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag, loss_val):
        path = self.output_dir / f"navigator_{tag}.pt"
        torch.save({
            "navigator_state": self.navigator.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": loss_val,
            "config": {
                "feat_dim":     self.navigator.feat_dim,
                "gru_hidden":   self.navigator.gru_hidden,
                "phi_dim":      self.navigator.phi_dim,
                "port_rank":    self.navigator.port_rank,
                "C_max":        self.navigator.C_max,
                "control_mode": self.navigator.control_mode,
                "dt_scale_min": self.navigator.dt_scale_min,
                "dt_scale_max": self.navigator.dt_scale_max,
            },
        }, path)
        print(f"  → Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.navigator.load_state_dict(ckpt["navigator_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Loaded checkpoint from {path}")