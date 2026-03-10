from __future__ import annotations

import torch
import torch.nn as nn


class FeatureProjector(nn.Module):
    def __init__(self, d_in: int, num_classes: int, hidden_dim: int = 64, latent_dim: int = 2,
                 identity_encoder: bool = False):
        super().__init__()
        self.identity_encoder = bool(identity_encoder and d_in == latent_dim)
        if self.identity_encoder:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Sequential(
                nn.Linear(d_in, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
        self.cls_head = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        logits = self.cls_head(z)
        return z, logits
