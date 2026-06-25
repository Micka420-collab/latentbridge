"""MiniGrid / BabyAI adapter — the bridge to a *public* benchmark.

BENCHMARKING.md §5: until the architecture runs on a recognized benchmark, you may
only say "promising on our task", not "better than SOTA". MiniGrid is the natural
first port — it is a goal-reaching gridworld with language missions (BabyAI), which
is exactly the regime the LatentBridge is built for, and it is light enough for CPU.

This wraps a gymnasium MiniGrid env into the same contract the rest of the codebase
expects (make_env / .reset / .step / .action_space / .obs_dim / .text_state() /
.true_goal_key() / .goal_state_text()), so EVERY method — random, PPO, DQN,
world-model, LatentBridge — runs on it unchanged via latentbridge.benchmark.

Notes:
  * FullyObsWrapper is used so the symbolic grid (incl. the goal) is in the
    observation — MiniGrid-Empty/DoorKey are "goal visible" tasks, i.e. a pure
    sample-efficiency contest. The language-bridge's distinct value needs the
    BabyAI GoTo* family (several candidate goals); that is a follow-up.
  * minigrid is imported lazily so importing this module without the package
    installed does not break the rest of the env package.
"""
from __future__ import annotations

import numpy as np

# MiniGrid cell encoding is [object_idx, color_idx, state]; object_idx maxes at
# ~10 (agent), color at ~5, state at ~3 — normalize by 11 to land in ~[0,1].
_ENC_SCALE = 11.0
_DIRS = {0: "right", 1: "down", 2: "left", 3: "up"}


class MiniGridAdapter:
    def __init__(self, env_id: str = "MiniGrid-Empty-5x5-v0", seed: int = 0,
                 max_steps: int | None = None, fully_obs: bool = True):
        import gymnasium as gym
        from minigrid.wrappers import FullyObsWrapper

        kwargs = {}
        if max_steps:
            kwargs["max_steps"] = int(max_steps)
        e = gym.make(env_id, **kwargs)
        if fully_obs:
            e = FullyObsWrapper(e)
        self.gym = e
        self.env_id = env_id
        self._seed = int(seed)
        self.action_space = int(e.action_space.n)
        obs, _ = e.reset(seed=self._seed)
        self._img_shape = obs["image"].shape
        self.obs_dim = int(np.prod(self._img_shape))
        self._obs = obs
        self.t = 0

    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        s = int(seed) if seed is not None else self._seed
        obs, _ = self.gym.reset(seed=s)
        self._obs = obs
        self.t = 0
        return self._flat(obs)

    def _flat(self, obs) -> np.ndarray:
        return (obs["image"].astype(np.float32) / _ENC_SCALE).reshape(-1)

    def step(self, action: int):
        obs, r, term, trunc, info = self.gym.step(int(action))
        self.t += 1
        self._obs = obs
        done = bool(term or trunc)
        reached = bool(term and r > 0)   # MiniGrid only gives reward>0 on the goal
        return self._flat(obs), float(r), done, {"reached": reached}

    # --- language interface (planner / bridge alignment target) -------- #
    @property
    def _u(self):
        return self.gym.unwrapped

    def text_state(self) -> str:
        x, y = self._u.agent_pos                 # (col, row)
        d = _DIRS.get(int(self._u.agent_dir), "?")
        return f"{self.env_id}. Agent at row {int(y)} col {int(x)} facing {d}."

    def goal_instruction(self) -> str:
        m = self._obs.get("mission") if isinstance(self._obs, dict) else None
        return str(m) if m else "Reach the green goal square."

    def _goal_cell(self):
        from minigrid.core.world_object import Goal
        g = self._u.grid
        for i in range(g.width):
            for j in range(g.height):
                c = g.get(i, j)
                if isinstance(c, Goal):
                    return (int(j), int(i))      # (row, col)
        return None

    def true_goal_key(self):
        cell = self._goal_cell()
        if cell is None:
            x, y = self._u.agent_pos
            return ("cell", int(y), int(x))
        return ("cell", cell[0], cell[1])

    def goal_state_text(self, key=None) -> str:
        if key is None:
            key = self.true_goal_key()
        _, r, c = key
        return f"{self.env_id}. Agent at row {r} col {c} facing right."
