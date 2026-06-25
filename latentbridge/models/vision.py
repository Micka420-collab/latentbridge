"""A small visual cortex (CNN) for pixel observations — étape 5.

Drop-in replacements for the MLP Encoder/Decoder when the observation is an image
(stacked frames). They take/return FLAT tensors (reshape internally) so the rest
of the pipeline — world model, bridge, losses — is unchanged. This is what lets
Shellia learn to SEE: spatial structure instead of a flat vector of pixels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, img_shape, latent_dim: int, hidden: int = 256):
        super().__init__()
        c, h, w = img_shape
        self.shape = img_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1), nn.SiLU(),   # 16->8
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),  # 8->4
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),  # 4->2
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, c, h, w)).shape[1]
        self.head = nn.Sequential(nn.Linear(n, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim))

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        c, h, w = self.shape
        return self.head(self.net(x_flat.view(-1, c, h, w)))


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, img_shape, hidden: int = 256):
        super().__init__()
        c, h, w = img_shape
        self.shape = img_shape
        self.pre = nn.Sequential(nn.Linear(latent_dim, 64 * 2 * 2), nn.SiLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.SiLU(),  # 2->4
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(),  # 4->8
            nn.ConvTranspose2d(32, c, 4, stride=2, padding=1),              # 8->16
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.pre(z).view(-1, 64, 2, 2)
        x = self.net(x)
        return x.reshape(x.shape[0], -1)   # flat, matches obs for recon MSE
