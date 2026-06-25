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


class RecurrentWorldModel(nn.Module):
    """GRU-augmented latent dynamics (a pragmatic RSSM) — opt-in via
    cfg.model.dynamics == "rssm".

    The memoryless residual MLP (WorldModel) compounds error fast on multi-step
    imagination (the desktop probe showed drift > RMS by ~5 steps). Here a
    recurrent state `g` is carried THROUGH the imagined rollout, giving the model
    memory of the trajectory so far — the lever the SOTA research (Dreamer-style
    RSSM) points to for long-horizon prediction.

    Deliberately kept compatible with the rest of the pipeline: z stays the
    per-obs latent (encoder output), so the bridge / decoder / align-ground-cycle
    losses are UNCHANGED. The recurrence lives only inside a rollout (init fresh
    per rollout), so there is NO cross-episode state to manage in eval/control.
    """

    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 128,
                 det_dim: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.det_dim = int(det_dim or hidden)
        self.act_embed = nn.Embedding(n_actions, hidden)
        self.gru = nn.GRUCell(latent_dim + hidden, self.det_dim)
        self.proj = nn.Sequential(nn.Linear(self.det_dim, hidden), nn.SiLU())
        self.delta_head = nn.Linear(hidden, latent_dim)
        self.reward_head = nn.Linear(hidden, 1)

    def init_state(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.det_dim, device=device)

    def forward(self, z: torch.Tensor, action: torch.Tensor, g: torch.Tensor = None):
        """One recurrent step. Returns (z_next, reward, g_next). g is created
        (zeros) if not provided — matches a fresh rollout start."""
        if g is None:
            g = self.init_state(z.shape[0], z.device)
        a = self.act_embed(action.long())
        g = self.gru(torch.cat([z, a], dim=-1), g)
        h = self.proj(g)
        z_next = z + self.delta_head(h)
        reward = self.reward_head(h).squeeze(-1)
        return z_next, reward, g
