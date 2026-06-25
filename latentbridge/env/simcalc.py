"""SimCalcWorld — a fast calculator simulator to train the LATENT BRIDGE that
then drives the REAL Windows Calculator (language -> real app).

Minimal, clean first version: the display starts CLEARED (0); the goal is a digit
given ONLY in language ("affichage N"); the solution is to press that digit. This
transfers to the real calculator (cleared -> press N -> shows N). The goal is
hidden from the observation, so the frozen-LLM -> bridge -> world-model channel is
the only way to know which digit to press. (Multi-digit/append dynamics are a
later extension.)
"""
from __future__ import annotations

import numpy as np

# actions: 0 = Clear, 1..10 = press digit 0..9
N_ACTIONS = 11


class SimCalcWorld:
    def __init__(self, max_value: int = 9, max_steps: int = 4, seed: int = 0,
                 dialect=None):
        self.max_value = int(max_value)
        self.max_steps = int(max_steps)
        self.action_space = N_ACTIONS
        # dialect = how a pressed digit maps to the display (a different "app");
        # None = identity (a normal calculator). Used by the few-shot experiment.
        self.dialect = list(dialect) if dialect is not None else None
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def obs_dim(self) -> int:
        return 10                                 # one-hot of the displayed digit

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.value = 0                            # cleared display
        self.goal = int(self.rng.integers(0, self.max_value + 1))
        self.t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        v = np.zeros(10, dtype=np.float32)
        v[int(np.clip(self.value, 0, 9))] = 1.0
        return v

    def set_value(self, v: int):                  # used by the real-calc adapter
        self.value = int(v)

    def step(self, action: int):
        a = int(action)
        if a == 0:
            self.value = 0                        # Clear
        else:
            d = a - 1                             # pressed digit
            self.value = self.dialect[d] if self.dialect else d   # app-specific mapping
        self.t += 1
        reached = (self.value == self.goal)
        reward = 1.0 if reached else -0.1
        done = reached or self.t >= self.max_steps
        return self._obs(), float(reward), bool(done), {"reached": reached}

    # --- language interface (shared vocab) --- #
    def text_state(self) -> str:
        return f"affichage {self.value}"

    def goal_instruction(self) -> str:
        return f"je veux le nombre {self.goal}"

    def true_goal_key(self):
        return self.goal

    def goal_menu(self):
        return [(str(n), f"afficher {n}") for n in range(self.max_value + 1)]

    def goal_state_text(self, key=None) -> str:
        n = self.goal if key is None else int(key)
        return f"affichage {n}"
