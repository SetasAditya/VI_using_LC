from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

from visualization.plots import adjusted_rand_index, normalized_mutual_info, purity_score, soft_assign_points


def ess_from_logw(logw) -> float:
    if torch.is_tensor(logw):
        w = torch.softmax(logw, dim=0).detach().cpu().numpy()
    else:
        lw = np.asarray(logw)
        lw = lw - lw.max()
        w = np.exp(lw)
        w = w / (w.sum() + 1e-8)
    return float(1.0 / np.sum(w ** 2))


def summarize_history(history: List[Dict], sigma: float) -> Dict[str, float]:
    purity, ari, nmi, ess, qacc, sens, transport, regret, eff_modes = [], [], [], [], [], [], [], [], []
    final_obs_freq = None
    final_obs_hist = None
    target_freq = None
    for item in history:
        batch_z = item["batch_z_true"].numpy() if hasattr(item["batch_z_true"], 'numpy') else np.asarray(item["batch_z_true"])
        batch_y = item["batch_y"].numpy() if hasattr(item["batch_y"], 'numpy') else np.asarray(item["batch_y"])
        centers = item["centers"].numpy() if hasattr(item["centers"], 'numpy') else np.asarray(item["centers"])
        pred = soft_assign_points(batch_z, centers, tau=0.7).argmax(axis=1)
        purity.append(purity_score(batch_y, pred))
        ari.append(adjusted_rand_index(batch_y, pred))
        nmi.append(normalized_mutual_info(batch_y, pred))
        ess.append(ess_from_logw(item["logw"]))
        qacc.append(float(item["selected_idx"] == item["teacher_idx"]))
        sens.append(float(item.get("sensitivity", math.nan)))
        ts = item["teacher_scores"].numpy() if hasattr(item["teacher_scores"], 'numpy') else np.asarray(item["teacher_scores"])
        regret.append(float(np.max(ts) - ts[int(item["selected_idx"])]))
        student_mu = item["particles"].mean(dim=0)
        teacher_mu = item["teacher_cloud"].mean(dim=0)
        transport.append(float(torch.norm(student_mu - teacher_mu).item()))
        em = item.get("effective_modes", math.nan)
        if torch.is_tensor(em):
            em = float(em.detach().cpu().item())
        eff_modes.append(float(em))
        if "observed_hist" in item:
            oh = item["observed_hist"].numpy() if hasattr(item["observed_hist"], 'numpy') else np.asarray(item["observed_hist"])
            final_obs_hist = oh.astype(float)
            final_obs_freq = final_obs_hist / max(final_obs_hist.sum(), 1.0)
        if "target_freq" in item:
            target_freq = item["target_freq"].numpy() if hasattr(item["target_freq"], 'numpy') else np.asarray(item["target_freq"])
    out = {
        "mean_purity": float(np.nanmean(purity)) if purity else math.nan,
        "mean_ari": float(np.nanmean(ari)) if ari else math.nan,
        "mean_nmi": float(np.nanmean(nmi)) if nmi else math.nan,
        "mean_ess": float(np.nanmean(ess)) if ess else math.nan,
        "mean_effective_modes": float(np.nanmean(eff_modes)) if eff_modes else math.nan,
        "query_accuracy": float(np.nanmean(qacc)) if qacc else math.nan,
        "mean_sensitivity": float(np.nanmean(sens)) if sens else math.nan,
        "mean_query_regret": float(np.nanmean(regret)) if regret else math.nan,
        "mean_transport_gap": float(np.nanmean(transport)) if transport else math.nan,
        "final_purity": float(purity[-1]) if purity else math.nan,
        "final_ari": float(ari[-1]) if ari else math.nan,
        "final_nmi": float(nmi[-1]) if nmi else math.nan,
        "final_ess": float(ess[-1]) if ess else math.nan,
        "final_effective_modes": float(eff_modes[-1]) if eff_modes else math.nan,
        "final_query_regret": float(regret[-1]) if regret else math.nan,
        "final_transport_gap": float(transport[-1]) if transport else math.nan,
    }
    if final_obs_freq is not None:
        out["final_observed_freq"] = [float(x) for x in final_obs_freq.tolist()]
        out["final_observed_hist"] = [float(x) for x in final_obs_hist.tolist()]
    if target_freq is not None:
        out["target_freq"] = [float(x) for x in target_freq.tolist()]
    return out
