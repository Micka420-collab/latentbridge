"""Non-learning policies + the shared held-out evaluator.

  - RandomPolicy   : the chance floor. Any method must beat this to mean anything.
  - LLMOnlyPolicy  : the planner proposes the goal, the agent goes straight to it
                     WITHOUT a learned world model. Measures what the world model
                     adds (BENCHMARKING.md §2: "LLM seul -> mesure l'apport du
                     world model"). Meaningful where a goal cell + simple dynamics
                     exist (gridworld); elsewhere it degrades to random and is
                     reported as such.

eval_policy mirrors evaluate._run_episodes exactly (same held-out seeds, same
success/steps/reward definition) so every method is scored on identical footing.
"""
from __future__ import annotations

import numpy as np

from ..env import make_env


class RandomPolicy:
    def __init__(self, n_actions: int, seed: int = 0):
        self.n = int(n_actions)
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env=None) -> int:
        n = env.action_space if env is not None else self.n
        return int(self.rng.integers(0, n))


class LLMOnlyPolicy:
    """Planner gives the goal; act greedily toward it with no learned dynamics."""

    def __init__(self, cfg, seed: int = 0):
        from ..llm import Planner
        self.planner = Planner(backend=cfg["planner"]["backend"],
                               model=cfg["planner"].get("model"),
                               size=cfg["env"].get("size", 6))
        self.rng = np.random.default_rng(seed)
        self.degraded = False  # set True the first time we fall back to random

    def act(self, obs, env=None) -> int:
        # Gridworld-style greedy: requires a (row,col) goal cell + 4 grid actions.
        if (env is not None and getattr(env, "action_space", None) == 4
                and hasattr(env, "agent")):
            try:
                sub = self.planner.propose(env)
                key = sub.get("goal_key")
                if key and key[0] == "cell":
                    _, gr, gc = key
                    r, c = env.agent
                    dr, dc = gr - r, gc - c
                    if dr != 0 and abs(dr) >= abs(dc):
                        return 1 if dr > 0 else 0   # down / up
                    if dc != 0:
                        return 3 if dc > 0 else 2   # right / left
                    return 0
            except Exception:
                pass
        self.degraded = True
        return int(self.rng.integers(0, env.action_space if env else 4))


def eval_policy(policy, cfg, device="cpu") -> dict:
    """Run ``policy`` on the held-out seeds (cfg.seed + 10_000 ..). Returns metrics."""
    base = cfg["seed"] + 10_000
    seeds = list(range(base, base + cfg["eval"]["episodes"]))
    succ, steps, rew = [], [], []
    for s in seeds:
        env = make_env(cfg, seed=s)
        obs = env.reset(seed=s)
        done, total, info = False, 0.0, {"reached": False}
        while not done:
            a = policy.act(obs, env=env)
            obs, r, done, info = env.step(a)
            total += r
        succ.append(1.0 if info.get("reached") else 0.0)
        steps.append(env.t)
        rew.append(total)
    return {
        "success_rate": float(np.mean(succ)),
        "avg_steps": float(np.mean(steps)),
        "avg_reward": float(np.mean(rew)),
    }
