"""Minimal dependency-free gridworld (the proving ground).

Observation: 3 x size x size float grid (channels: agent, goal, wall), flattened.
Actions: 0=up 1=down 2=left 3=right.
Reward: +1 on reaching goal, small step penalty, optional shaping handled outside.
Generalization: agent/goal positions are randomized per episode, so a held-out
set of seeds tests whether the model learned dynamics rather than memorized layouts.
"""
from __future__ import annotations

import numpy as np

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}


class GridWorld:
    def __init__(self, size: int = 6, max_steps: int = 50, n_walls: int = 0,
                 step_penalty: float = 0.01, include_goal_in_obs: bool = False,
                 seed: int = 0):
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.n_walls = int(n_walls)
        self.step_penalty = float(step_penalty)
        # When False, the goal is NOT visible in the observation — it is only
        # communicated to the agent in language. The agent then *must* use the
        # LLM -> bridge -> world-model channel to know where to go. This is what
        # makes the bridge's value measurable instead of redundant.
        self.include_goal_in_obs = bool(include_goal_in_obs)
        self.action_space = 4
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def n_channels(self) -> int:
        return 3 if self.include_goal_in_obs else 2  # agent, [goal], wall

    @property
    def obs_dim(self) -> int:
        return self.n_channels * self.size * self.size

    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        self.rng.shuffle(cells)
        self.walls = set(cells[: self.n_walls])
        free = [c for c in cells if c not in self.walls]
        self.agent = free[0]
        self.goal = free[1]
        self.t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        g = np.zeros((self.n_channels, self.size, self.size), dtype=np.float32)
        g[0, self.agent[0], self.agent[1]] = 1.0
        wall_ch = 2 if self.include_goal_in_obs else 1
        if self.include_goal_in_obs:
            g[1, self.goal[0], self.goal[1]] = 1.0
        for (r, c) in self.walls:
            g[wall_ch, r, c] = 1.0
        return g.reshape(-1)

    def step(self, action: int):
        dr, dc = ACTIONS[int(action)]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.walls:
            self.agent = (nr, nc)
        self.t += 1
        done = self.agent == self.goal
        reached = done
        reward = 1.0 if reached else -self.step_penalty
        if self.t >= self.max_steps:
            done = True
        return self._obs(), float(reward), bool(done), {"reached": reached}

    # --- language interface (for the LLM planner / bridge alignment target) --- #
    def text_state(self) -> str:
        # Describes only the OBSERVABLE state (agent position). The goal is an
        # instruction, not part of the state, so it is deliberately absent here.
        return f"Grid {self.size}x{self.size}. Agent at row {self.agent[0]} col {self.agent[1]}."

    def goal_instruction(self) -> str:
        return f"Reach the goal at row {self.goal[0]} col {self.goal[1]}."

    def true_goal_key(self):
        return ("cell", int(self.goal[0]), int(self.goal[1]))

    def goal_state_text(self, key=None) -> str:
        # the imagined goal state ("agent at the goal cell"), in text_state format
        if key is None:
            key = self.true_goal_key()
        _, r, c = key
        return f"Grid {self.size}x{self.size}. Agent at row {r} col {c}."

    def manhattan(self) -> int:
        return abs(self.agent[0] - self.goal[0]) + abs(self.agent[1] - self.goal[1])

    @property
    def goal_xy(self):
        return self.goal
