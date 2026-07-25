"""Model-predictive controller over the learned world model.

This is where the three pieces meet:
  - world model -> imagined rollouts to score action sequences
  - LLM planner -> proposes a subgoal (when guided=True)
  - bridge      -> turns the LLM's language subgoal into a latent target the
                   world model can be steered toward

Comparing guided vs unguided success rate is the experiment's evidence that the
LLM<->world-model coupling actually helps.

Two trajectory optimizers, selected by cfg.control.optimizer:
  shooting  sample n_samples uniform-random action sequences, pick the best
  cem       cross-entropy method: iteratively refit a per-timestep categorical
            distribution on the elite sequences. Uses the SAME total rollout
            budget as shooting (n_samples split across cem_iters iterations),
            so the two are compute-comparable.
"""
from __future__ import annotations

import collections

import numpy as np
import torch
import torch.nn.functional as F


class MPCController:
    def __init__(self, encoder, world_model, bridge=None, featurizer=None,
                 planner=None, horizon: int = 5, n_samples: int = 256,
                 gamma: float = 0.95, guide_weight: float = 1.0,
                 optimizer: str = "shooting", cem_iters: int = 4,
                 cem_elite_frac: float = 0.125, cem_alpha: float = 0.7,
                 revisit_weight: float = 0.0, revisit_mem: int = 6,
                 seed: int = 0, device="cpu"):
        if optimizer not in ("shooting", "cem"):
            raise ValueError(f"unknown control optimizer: {optimizer}")
        self.encoder = encoder
        self.world_model = world_model
        self.bridge = bridge
        self.featurizer = featurizer
        self.planner = planner
        self.horizon = horizon
        self.n_samples = n_samples
        self.gamma = gamma
        self.guide_weight = guide_weight
        self.optimizer = optimizer
        self.cem_iters = int(cem_iters)
        self.cem_elite_frac = float(cem_elite_frac)
        self.cem_alpha = float(cem_alpha)
        # Anti-cycling: guided failures are near-goal 2-cycles (the replanned
        # argmax flips forever where the guidance field is locally flat), so a
        # short memory of VISITED latents repels imagined rollouts from recently
        # occupied states. Guided-only; 0.0 = exact previous behavior.
        self.revisit_weight = float(revisit_weight)
        self.revisit_mem = int(revisit_mem)
        self._visited: collections.deque = collections.deque(maxlen=self.revisit_mem)
        self.device = device
        self.n_actions = world_model.n_actions
        # controller-owned RNG: action sampling must not depend on (or disturb)
        # the global torch RNG state, so repeated evaluations are reproducible
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(int(seed))
        # subgoal text -> target latent memo. The planner proposes the same
        # goal at every step of an episode, so embedding + from_lang would
        # otherwise be recomputed each act() call.
        self._target_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def act(self, obs: np.ndarray, env=None, guided: bool = False) -> int:
        z0 = self.encoder(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        if env is not None and getattr(env, "t", None) == 0:
            self._visited.clear()   # fresh episode -> fresh revisit memory

        target_latent = None
        if guided and self.planner is not None and self.bridge is not None:
            sub = self.planner.propose(env)
            text = sub["subgoal_text"]
            target_latent = self._target_cache.get(text)
            if target_latent is None:
                emb = self.featurizer.embed(text)
                emb_t = torch.as_tensor(emb, dtype=torch.float32, device=self.device).unsqueeze(0)
                target_latent = self.bridge.from_lang(emb_t)  # (1, D)
                if len(self._target_cache) > 4096:
                    self._target_cache.clear()
                self._target_cache[text] = target_latent

        if guided and self.revisit_weight > 0:
            self._visited.append(z0.squeeze(0))

        if self.optimizer == "cem":
            return self._plan_cem(z0, target_latent)
        return self._plan_shooting(z0, target_latent)

    # ------------------------------------------------------------------ #
    def _score(self, z0: torch.Tensor, actions: torch.Tensor,
               target_latent: torch.Tensor | None) -> torch.Tensor:
        """Imagined discounted return of each action sequence (K, H) -> (K,).
        When a target latent is given, a dense discounted distance-to-goal
        shaping steers the whole rollout toward the LLM-imagined goal (the
        bridge's contribution to control), and (when revisit_weight > 0) a
        proximity penalty to recently-visited latents breaks replanning cycles."""
        K, H = actions.shape
        z_cur = z0.expand(K, -1).contiguous()
        recurrent = hasattr(self.world_model, "init_state")
        g = self.world_model.init_state(K, self.device) if recurrent else None
        M = (torch.stack(list(self._visited))
             if (target_latent is not None and self.revisit_weight > 0
                 and self._visited) else None)
        total_r = torch.zeros(K, device=self.device)
        for t in range(H):
            if recurrent:
                z_cur, r, g = self.world_model(z_cur, actions[:, t], g)
            else:
                z_cur, r = self.world_model(z_cur, actions[:, t])
            total_r += (self.gamma ** t) * r
            if target_latent is not None:
                dist = ((z_cur - target_latent) ** 2).mean(-1)
                total_r -= self.guide_weight * (self.gamma ** t) * dist
                if M is not None:
                    # self-scaling proximity to recently visited latents: the
                    # d2.mean()/dist.mean() normalizations make one weight
                    # transfer across envs and latent scales
                    d2 = ((z_cur.unsqueeze(1) - M.unsqueeze(0)) ** 2).mean(-1)
                    prox = torch.exp(-d2 / (d2.mean() + 1e-8)).max(1).values
                    total_r -= (self.revisit_weight * self.guide_weight
                                * (self.gamma ** t) * prox * dist.mean())
        return total_r

    def _plan_shooting(self, z0, target_latent) -> int:
        K, H = self.n_samples, self.horizon
        actions = torch.randint(0, self.n_actions, (K, H),
                                generator=self._gen, device=self.device)
        scores = self._score(z0, actions, target_latent)
        best = int(torch.argmax(scores).item())
        return int(actions[best, 0].item())

    def _plan_cem(self, z0, target_latent) -> int:
        H, A = self.horizon, self.n_actions
        pop = max(2, self.n_samples // self.cem_iters)   # equal total budget
        n_elite = max(1, int(round(pop * self.cem_elite_frac)))
        probs = torch.full((H, A), 1.0 / A, device=self.device)
        best_seq, best_score = None, -float("inf")
        for _ in range(self.cem_iters):
            actions = torch.multinomial(probs, pop, replacement=True,
                                        generator=self._gen).T.contiguous()  # (pop, H)
            scores = self._score(z0, actions, target_latent)
            elite_idx = torch.topk(scores, n_elite).indices
            if float(scores[elite_idx[0]]) > best_score:
                best_score = float(scores[elite_idx[0]])
                best_seq = actions[elite_idx[0]].clone()
            elite_freq = F.one_hot(actions[elite_idx], A).float().mean(0)  # (H, A)
            probs = self.cem_alpha * elite_freq + (1.0 - self.cem_alpha) * probs
            # keep every action reachable so the distribution can recover
            probs = probs.clamp_min(1e-6)
            probs = probs / probs.sum(-1, keepdim=True)
        return int(best_seq[0].item())
