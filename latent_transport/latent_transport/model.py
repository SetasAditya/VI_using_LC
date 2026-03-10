from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetContextEncoder(nn.Module):
    def __init__(self, hidden: int, context_dim: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, context_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * context_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, context_dim),
        )

    def _pool(self, pts: torch.Tensor, role: float) -> torch.Tensor:
        role_col = torch.full((pts.shape[0], pts.shape[1], 1), role, dtype=pts.dtype, device=pts.device)
        return self.phi(torch.cat([pts, role_col], dim=-1)).mean(dim=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, step_frac: torch.Tensor) -> torch.Tensor:
        return self.rho(
            torch.cat(
                [self._pool(src, 0.0), self._pool(tgt, 1.0), step_frac.squeeze(1)],
                dim=-1,
            )
        )


class ParticleEncoder(nn.Module):
    def __init__(self, hidden: int, context_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_rep = c[:, None, :].expand(-1, x.shape[1], -1)
        return self.net(torch.cat([x, c_rep], dim=-1))


class ParticleDecoder(nn.Module):
    def __init__(self, hidden: int, context_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_rep = c[:, None, :].expand(-1, z.shape[1], -1)
        return self.net(torch.cat([z, c_rep], dim=-1))


class LatentPHTransport(nn.Module):
    def __init__(self, hidden: int, context_dim: int, latent_dim: int):
        super().__init__()
        self.context_net = SetContextEncoder(hidden, context_dim)
        self.encoder = ParticleEncoder(hidden, context_dim, latent_dim)
        self.decoder = ParticleDecoder(hidden, context_dim, latent_dim)

        self.gradU = nn.Sequential(
            nn.Linear(latent_dim + 1 + context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        self.control = nn.Sequential(
            nn.Linear(2 * latent_dim + 1 + context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
            nn.Tanh(),
        )
        self.damping = nn.Sequential(
            nn.Linear(1 + context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def context(self, src, tgt, step_frac):
        return self.context_net(src, tgt, step_frac)

    def encode(self, x, c):
        return self.encoder(x, c)

    def decode(self, z, c):
        return self.decoder(z, c)

    def field(self, z, r, tau, c) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_u = self.gradU(torch.cat([z, tau, c], dim=-1))
        damp = F.softplus(self.damping(torch.cat([tau, c], dim=-1))) + 0.01
        u = 1.5 * self.control(torch.cat([z, r, tau, c], dim=-1))
        dz = r
        dr = -grad_u - damp * r + u
        return dz, dr, u

    def rollout_latent(self, z0, c, n_steps: int):
        z = z0.clone()
        r = torch.zeros_like(z)
        dt = 1.0 / n_steps
        act_sq = 0.0

        for s in range(n_steps):
            tau = torch.full(
                (z.shape[0], z.shape[1], 1),
                (s + 0.5) / n_steps,
                device=z.device,
                dtype=z.dtype,
            )
            c_rep = c[:, None, :].expand(-1, z.shape[1], -1)
            dz, dr, u = self.field(z, r, tau, c_rep)
            r = r + dt * dr
            z = z + dt * dz
            act_sq = act_sq + u.pow(2).mean()

        return z, act_sq / n_steps

    def rollout_decoded(self, src, tgt, step_frac, n_steps: int):
        c = self.context(src, tgt, step_frac)
        z0 = self.encode(src, c)
        zT, act = self.rollout_latent(z0, c, n_steps)

        x0_hat = self.decode(z0, c)
        xT_hat = self.decode(zT, c)
        xT = src + (xT_hat - x0_hat)

        return xT, zT, c, act


def cubic_bridge(z0: torch.Tensor, z1: torch.Tensor, tau: torch.Tensor):
    h0 = 2 * tau**3 - 3 * tau**2 + 1
    h1 = -2 * tau**3 + 3 * tau**2
    dh0 = 6 * tau**2 - 6 * tau
    dh1 = -6 * tau**2 + 6 * tau
    ddh0 = 12 * tau - 6
    ddh1 = -12 * tau + 6
    return h0 * z0 + h1 * z1, dh0 * z0 + dh1 * z1, ddh0 * z0 + ddh1 * z1