from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentEnergy(nn.Module):
    def __init__(self, ctx_dim: int, hidden_dim: int = 64, latent_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + ctx_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0).expand(z.shape[0], -1)
        return self.net(torch.cat([z, ctx], dim=-1)).squeeze(-1)


class LevelSetMixture(nn.Module):
    """Positive latent level function built as a sum of anisotropic Gaussian kernels.

    This is the differentiable analogue of the Astolfi superlevel clustering object.
    The context only induces small center shifts and weight corrections; the base geometry
    remains explicit and interpretable through centers / scales / amplitudes.
    """

    def __init__(self, latent_dim: int = 2, num_components: int = 8, ctx_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.base_centers = nn.Parameter(torch.randn(num_components, latent_dim) * 2.0)
        self.raw_scales = nn.Parameter(torch.zeros(num_components, latent_dim))
        self.raw_logits = nn.Parameter(torch.zeros(num_components))
        self.ctx_shift = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_components * latent_dim),
        )
        self.ctx_logits = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_components),
        )

    def component_params(self, ctx: torch.Tensor | None = None):
        centers = self.base_centers
        logits = self.raw_logits
        if ctx is not None:
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            shift = 0.25 * torch.tanh(self.ctx_shift(ctx)).mean(dim=0).view(self.num_components, self.latent_dim)
            dlogits = 0.25 * torch.tanh(self.ctx_logits(ctx)).mean(dim=0)
            centers = centers + shift
            logits = logits + dlogits
        scales = F.softplus(self.raw_scales) + 0.35
        weights = torch.softmax(logits, dim=0)
        return centers, scales, weights

    def forward(self, z: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        centers, scales, weights = self.component_params(ctx)
        diff = (z[:, None, :] - centers[None, :, :]) / scales[None, :, :]
        quad = 0.5 * (diff ** 2).sum(dim=-1)
        ker = torch.exp(-quad)
        return ker @ weights + 1e-8

    def log_density(self, z: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        return torch.log(self.forward(z, ctx).clamp_min(1e-8))


class KineticEnergy(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.raw_mass = nn.Parameter(torch.zeros(latent_dim))

    def mass(self) -> torch.Tensor:
        return F.softplus(self.raw_mass) + 0.25

    def velocity(self, p: torch.Tensor) -> torch.Tensor:
        return p / self.mass()[None, :]

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        vel = self.velocity(p)
        return 0.5 * (p * vel).sum(dim=-1)


class MemoryPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRUCell(in_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, feat: torch.Tensor, h: torch.Tensor):
        h_new = self.gru(feat.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        raw = self.head(h_new)
        ctrl = {
            "dt": 0.035 + 0.045 * torch.sigmoid(raw[0]),
            "gamma": 0.10 + 0.14 * torch.sigmoid(raw[1]),
            "obs_gain": 0.22 + 0.52 * torch.sigmoid(raw[2]),
            "topo_gain": 0.04 + 0.20 * torch.sigmoid(raw[3]),
        }
        return ctrl, h_new


class QueryPolicy(nn.Module):
    def __init__(self, state_dim: int, cand_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(state_dim + cand_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_ctx: torch.Tensor, cand_feats: torch.Tensor) -> torch.Tensor:
        s = state_ctx.unsqueeze(0).expand(cand_feats.shape[0], -1)
        return self.score(torch.cat([s, cand_feats], dim=-1)).squeeze(-1)
