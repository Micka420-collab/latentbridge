"""Residual Decoder with skip connections.

latent z -> Linear -> SiLU -> ResidualBlock -> ResidualBlock -> obs

Mirrors the encoder architecture for symmetric grounding.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Linear -> SiLU -> Linear + skip, with pre-LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class ResidualDecoder(nn.Module):
    """latent z -> reconstructed obs."""

    def __init__(self, latent_dim: int, obs_dim: int, hidden: int = 256,
                 num_blocks: int = 2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden, obs_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(z)
        h = self.blocks(h)
        return self.output_proj(h)
