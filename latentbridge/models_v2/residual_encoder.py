"""Residual Encoder with skip connections and LayerNorm.

obs -> Linear -> SiLU -> LayerNorm -> ResidualBlock -> ResidualBlock -> latent z

Why: plain MLP suffers from vanishing gradients and poor conditioning
on gridworld (spatial relationships). Residual connections let us go
deeper while keeping gradients healthy, and LayerNorm stabilises training.
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


class ResidualEncoder(nn.Module):
    """obs -> latent z with residual blocks."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden: int = 256,
                 num_blocks: int = 2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden, latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(obs)
        h = self.blocks(h)
        return self.output_proj(h)
