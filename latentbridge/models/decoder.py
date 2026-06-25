import torch
import torch.nn as nn


class Decoder(nn.Module):
    """latent z -> reconstructed obs (grounds the latent in observations)."""

    def __init__(self, latent_dim: int, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
