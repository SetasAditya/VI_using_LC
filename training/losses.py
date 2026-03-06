"""
Training Losses for GMM SPHS Episodic Training.

Combined loss:
    L = lambda_terminal * L_terminal
      + lambda_topo     * L_topo
      + lambda_ess      * L_ess
      + lambda_casimir  * L_casimir

All losses are designed to flow gradients through the BAOAB steps
(phi depends on navigator outputs via the transport dynamics).
PHC topology features are stop-gradded (non-differentiable).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from topology.diagnostics import TopoDiagnostics
from data.gmm_problem import GMMProblem, unpack_phi
from dynamics.canonicalize import hungarian_match


# ──────────────────────────────────────────────────────────────────────────────
# Terminal loss (requires ground truth)
# ──────────────────────────────────────────────────────────────────────────────

def terminal_loss(
    phi: torch.Tensor,          # [M, phi_dim] final particles
    problem: GMMProblem,        # ground truth
    weights: torch.Tensor,      # [M] normalized weights
) -> Dict[str, torch.Tensor]:
    """
    Weighted posterior quality loss with Huber-clamped mse_mu.

    Uses Huber loss on the mean error rather than squared error to
    prevent explosive gradients when particles are far from the truth
    early in training (mse_mu can reach 100+ without Huber).

    Returns dict with all components and 'total'.
    """
    K, D = problem.K, problem.D

    pi_tilde, mu, L_vecs = unpack_phi(phi, K, D)  # [M,K], [M,K,D], [M,K,chol]

    w = weights.detach()   # stop-grad weights

    # ── Huber-MSE_mu: Hungarian-matched mean error ────────────────────────────
    mu_weighted = (w.unsqueeze(-1).unsqueeze(-1) * mu).sum(0)  # [K, D]

    try:
        perm = hungarian_match(mu_weighted.detach(), problem.mu_true)
        # Apply perm to PREDICTION (not ground truth) — consistent with eval
        mu_weighted_matched = mu_weighted[perm]
    except Exception:
        mu_weighted_matched = mu_weighted
        perm = torch.arange(K, device=phi.device)

    # Huber loss on permuted prediction vs fixed ground truth
    mu_err = mu_weighted_matched - problem.mu_true                  # [K, D]
    delta = 2.0
    huber_mu = F.huber_loss(mu_weighted_matched, problem.mu_true, delta=delta, reduction='mean')

    # Raw MSE for logging (not used in gradient)
    mse_mu = (mu_err ** 2).mean().detach()

    # ── MSE_Sigma ─────────────────────────────────────────────────────────────
    from data.gmm_problem import L_vec_to_matrix
    L = L_vec_to_matrix(L_vecs, D)                              # [M, K, D, D]
    Sigma = torch.bmm(
        L.reshape(-1, D, D), L.reshape(-1, D, D).transpose(-1, -2),
    ).reshape(phi.shape[0], K, D, D)
    Sigma_weighted = (w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * Sigma).sum(0)

    try:
        # Apply same perm to sigma prediction
        Sigma_weighted_matched = Sigma_weighted[perm]
    except Exception:
        Sigma_weighted_matched = Sigma_weighted

    mse_sigma = ((Sigma_weighted_matched - problem.Sigma_true) ** 2).mean()
    # Clamp sigma loss — covariance errors are in squared units, can be huge
    mse_sigma_loss = mse_sigma.clamp(max=50.0)

    # ── Weighted NLL: differentiable dense signal ─────────────────────────────
    from dynamics.gmm_energy import gmm_log_likelihood
    log_lik = gmm_log_likelihood(phi, problem.X, K, D)              # [M]
    # Normalize by N to make scale independent of batch size
    N = problem.X.shape[0]
    nll = -(w * log_lik).sum() / N
    nll_loss = nll.clamp(max=20.0)   # cap at 20 nats / data point

    # ── Combined ──────────────────────────────────────────────────────────────
    total = huber_mu + 0.1 * mse_sigma_loss + 0.01 * nll_loss

    return {
        "mse_mu":     mse_mu,           # raw (for logging)
        "huber_mu":   huber_mu,         # used in gradient
        "mse_sigma":  mse_sigma,
        "energy":     nll,
        "total":      total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Topology loss
# ──────────────────────────────────────────────────────────────────────────────

def topology_loss(
    topo_history: List[TopoDiagnostics],
    K_true: int,
    lambda_collapse: float = 1.0,
    lambda_hw: float = 0.5,
    lambda_ess: float = 0.3,
) -> torch.Tensor:
    """
    Topology loss over episode history.

    Penalizes:
        - Mode collapse: |C_tau - K_true|
        - Low mass entropy H_W
        - Low min per-cluster ESS

    NOTE: topology features are stop-gradded — this loss acts as a
    supervision signal for the navigator but doesn't flow through PHC.

    Returns: scalar loss
    """
    loss = torch.tensor(0.0)

    for topo in topo_history:
        # Mode count penalty
        C_diff = abs(topo.C_tau - K_true)
        loss = loss + lambda_collapse * C_diff

        # Entropy penalty (want high H_W → balanced coverage)
        H_W_max = math.log(max(K_true, 1))
        loss = loss + lambda_hw * max(0.0, H_W_max * 0.5 - topo.H_W)

        # ESS penalty
        if topo.C_tau > 1:
            loss = loss + lambda_ess * max(0.0, 5.0 - topo.min_ESS_c)

    return loss / max(len(topo_history), 1)


# ──────────────────────────────────────────────────────────────────────────────
# ESS loss (differentiable via phi)
# ──────────────────────────────────────────────────────────────────────────────

def ess_loss(
    phi: torch.Tensor,          # [M, phi_dim] particles
    X: torch.Tensor,            # [N, D] data
    energy_fn,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable ESS loss: -log(ESS / M).

    Encourages particles to have similar free energy (uniform weights).
    This IS differentiable w.r.t. phi.

    Returns: scalar loss
    """
    U = energy_fn.free_energy(phi, X)           # [M]
    log_w = -beta * U
    log_w = log_w - log_w.max()                 # stabilize
    w = torch.softmax(log_w, dim=0)             # [M]

    ess = (w.sum() ** 2) / (w ** 2).sum().clamp(min=1e-30)
    M = phi.shape[0]

    return -torch.log(ess / M + 1e-8)          # minimize → maximize ESS


# ──────────────────────────────────────────────────────────────────────────────
# Combined loss
# ──────────────────────────────────────────────────────────────────────────────

def combined_episode_loss(
    phi_final: torch.Tensor,
    weights: torch.Tensor,
    topo_history: List[TopoDiagnostics],
    casimir_result,
    problem: GMMProblem,
    energy_fn,
    lambda_terminal: float = 1.0,
    lambda_topo: float = 0.1,
    lambda_ess: float = 0.1,
    lambda_casimir: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """
    Full episode loss combining all components.

    Args:
        phi_final:     [M, phi_dim] terminal particles
        weights:       [M] terminal SMC weights
        topo_history:  list of TopoDiagnostics across batches
        casimir_result: CasimirResult from last step
        problem:       GMMProblem with ground truth
        energy_fn:     GMMEnergy instance
        lambda_*:      loss weights
    Returns:
        dict with all loss components + 'total'
    """
    losses = {}

    # Terminal loss
    term = terminal_loss(phi_final, problem, weights)
    losses["mse_mu"]    = term["mse_mu"]
    losses["mse_sigma"] = term["mse_sigma"]
    losses["energy"]    = term["energy"]
    L_terminal = term["total"]

    # Topology loss (stop-grad)
    L_topo = topology_loss(topo_history, problem.K)

    # ESS loss (differentiable)
    L_ess = ess_loss(phi_final, problem.X, energy_fn)

    # Casimir loss (differentiable structural penalty)
    from fidelity.casimir import CasimirChecker
    casimir_checker = CasimirChecker(problem.K, problem.D)
    L_casimir = casimir_checker.casimir_loss(phi_final)

    total = (
        lambda_terminal * L_terminal +
        lambda_topo     * L_topo +
        lambda_ess      * L_ess +
        lambda_casimir  * L_casimir
    )

    losses.update({
        "terminal": L_terminal,
        "topo":     L_topo,
        "ess":      L_ess,
        "casimir":  L_casimir,
        "total":    total,
    })

    return losses
