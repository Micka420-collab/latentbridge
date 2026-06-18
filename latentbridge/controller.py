"""Model-predictive controller (random shooting) over the learned world model.

This is where the three pieces meet:
  - world model -> imagined rollouts to score action sequences
  - LLM planner -> proposes a subgoal (when guided=True)
  - bridge      -> turns the LLM's language subgoal into a latent target the
                   world model can be steered toward

Comparing guided vs unguided success rate is the experiment's evidence that the
LLM<->world-model coupling actually helps.
"""
from __future__ import annotations

import numpy as np
import torch


class MPCController:
    def __init__(self, encoder, world_model, bridge=None, featurizer=None,
                 planner=None, horizon: int = 5, n_samples: int = 256,
                 gamma: float = 0.95, guide_weight: float = 1.0, device="cpu"):
        self.encoder = encoder
        self.world_model = world_model
        self.bridge = bridge
        self.featurizer = featurizer
        self.planner = planner
        self.horizon = horizon
        self.n_samples = n_samples
        self.gamma = gamma
        self.guide_weight = guide_weight
        self.device = device
        self.n_actions = world_model.n_actions

    @torch.no_grad()
    def act(self, obs: np.ndarray, env=None, guided: bool = False) -> int:
        z0 = self.encoder(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))

        target_latent = None
        if guided and self.planner is not None and self.bridge is not None:
            sub = self.planner.propose(env)
            emb = self.featurizer.embed(sub["subgoal_text"])
            emb_t = torch.as_tensor(emb, dtype=torch.float32, device=self.device).unsqueeze(0)
            target_latent = self.bridge.from_lang(emb_t)  # (1, D)

        # random shooting: sample action sequences, roll out, score
        K, H = self.n_samples, self.horizon
        actions = torch.randint(0, self.n_actions, (K, H), device=self.device)
        z = z0.expand(K, -1).contiguous()
        discounts = (self.gamma ** torch.arange(H, device=self.device)).unsqueeze(0)

        total_r = torch.zeros(K, device=self.device)
        z_cur = z
        for t in range(H):
            z_cur, r = self.world_model(z_cur, actions[:, t])
            total_r += (self.gamma ** t) * r
            if target_latent is not None:
                # dense, discounted shaping: steer the whole rollout toward the
                # LLM-imagined goal latent (the bridge's contribution to control)
                dist = ((z_cur - target_latent) ** 2).mean(-1)
                total_r -= self.guide_weight * (self.gamma ** t) * dist

        best = int(torch.argmax(total_r).item())
        return int(actions[best, 0].item())
