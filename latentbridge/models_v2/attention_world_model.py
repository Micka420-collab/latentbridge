"""Attention-based World Model for richer dynamics learning.

(z_t, action) -> Residual MLP -> Multi-head Self-Attention pool -> delta_z, reward

The self-attention layer lets the model attend across different parts of
the latent representation — critical for gridworld where spatial relationships
(up/down/left/right) interact. Residual delta prediction keeps training stable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionWorldModel(nn.Module):
    """Latent dynamics with attention-augmented trunk.

    Architecture:
      input = concat(z, one_hot(action))
      trunk = Linear -> SiLU -> LayerNorm -> Linear -> SiLU -> LayerNorm
      attended = trunk + MultiheadSelfAttention(trunk)   # residual attention
      delta_z = delta_head(attended)
      reward = reward_head(attended)
      z_next = z + delta_z
    """

    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 256,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions

        in_dim = latent_dim + n_actions
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        # Self-attention on the hidden representation.
        # We treat the hidden vector as a sequence of num_heads "tokens"
        # by reshaping: (B, hidden) -> (B, num_heads, head_dim)
        assert hidden % num_heads == 0, f"hidden ({hidden}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=self.head_dim,
            num_heads=1,  # single head per "token", the multi-head is from reshaping
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden)

        self.delta_head = nn.Linear(hidden, latent_dim)
        self.reward_head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, action: torch.Tensor):
        a = F.one_hot(action.long(), self.n_actions).float()
        x = torch.cat([z, a], dim=-1)

        h = self.input_proj(x)
        trunk_out = self.trunk(h)

        # Reshape for multi-head attention: (B, hidden) -> (B, num_heads, head_dim)
        B = trunk_out.shape[0]
        tokens = trunk_out.view(B, self.num_heads, self.head_dim)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out.reshape(B, -1)  # (B, hidden)

        # Residual attention
        attended = self.attn_norm(trunk_out + self.dropout(attn_out))

        delta = self.delta_head(attended)
        reward = self.reward_head(attended).squeeze(-1)
        return z + delta, reward

    @torch.no_grad()
    def rollout(self, z: torch.Tensor, actions: torch.Tensor):
        """actions: (B, H). Returns predicted latents (B, H, D) and rewards (B, H)."""
        zs, rs = [], []
        for t in range(actions.shape[1]):
            z, r = self.forward(z, actions[:, t])
            zs.append(z)
            rs.append(r)
        return torch.stack(zs, dim=1), torch.stack(rs, dim=1)
