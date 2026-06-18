import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """Latent dynamics: (z_t, action) -> z_{t+1}, reward.

    Predicts a residual to z_t (stabler) plus a scalar reward. Supports
    multi-step "imagined" rollouts by feeding predicted latents back in,
    which is what the planner uses for model-predictive control.
    """

    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(hidden, latent_dim)
        self.reward_head = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor, action: torch.Tensor):
        a = F.one_hot(action.long(), self.n_actions).float()
        h = self.trunk(torch.cat([z, a], dim=-1))
        z_next = z + self.delta_head(h)
        reward = self.reward_head(h).squeeze(-1)
        return z_next, reward

    @torch.no_grad()
    def rollout(self, z: torch.Tensor, actions: torch.Tensor):
        """actions: (B, H). Returns predicted latents (B, H, D) and rewards (B, H)."""
        zs, rs = [], []
        for t in range(actions.shape[1]):
            z, r = self.forward(z, actions[:, t])
            zs.append(z)
            rs.append(r)
        return torch.stack(zs, dim=1), torch.stack(rs, dim=1)
