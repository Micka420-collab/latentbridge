"""LifeWorld — a synthetic model of *your day*.

Same interface as GridWorld, so the entire LatentBridge pipeline (train, eval,
bridge, MPC) works unchanged. Here the "world model" learns how a day evolves:
choosing activities advances time, spends/restores energy, completes tasks, and
ticks off habits. The DAILY INTENTION (what you want out of the day) is the goal
— given ONLY in language, never in the observation — so the assistant must use
the LLM -> bridge -> world-model channel to plan a day that fits the intention.

This is the bridge from "toy proving ground" to "personal assistant": swap the
gridworld for the rhythms of a real life, keep the exact same architecture.
"""
from __future__ import annotations

import numpy as np

HABITS = ["exercise", "study", "social"]
# action index -> activity name
ACTIVITIES = ["deep_work", "light_work", "exercise", "study", "social", "rest"]

# goal_key -> (natural-language intention, short menu description)
GOALS = {
    "productive": ("I really need to get my important work done today.",
                   "finish the important / high-priority tasks"),
    "balanced":   ("I want a balanced day: move my body, study, and see people.",
                   "do all three habits (exercise, study, social)"),
    "recovery":   ("I'm exhausted, I just want to rest and recharge today.",
                   "rest and keep energy high"),
}


def _bucket(e):
    return "high" if e > 0.66 else ("low" if e < 0.33 else "medium")


class LifeWorld:
    def __init__(self, n_slots: int = 10, max_tasks: int = 4, n_high: int = 2,
                 seed: int = 0):
        self.n_slots = int(n_slots)
        self.max_tasks = int(max_tasks)
        self.n_high = int(n_high)
        self.action_space = len(ACTIVITIES)
        self.max_steps = self.n_slots
        self.rng = np.random.default_rng(seed)
        self.reset()

    @classmethod
    def from_data(cls, data: dict, n_slots: int = 10, max_tasks: int = 4, seed: int = 0):
        """Build a day from the user's REAL data (titles/priorities/habits/energy).
        Dimensions (n_slots, max_tasks) must match the trained checkpoint."""
        def is_high(t):
            return str(t.get("priority", "low")).lower() in ("high", "haute", "1", "true", "yes")
        tasks_in = sorted(data.get("tasks", []), key=lambda t: 0 if is_high(t) else 1)[:max_tasks]
        n_high = sum(1 for t in tasks_in if is_high(t))
        self = cls(n_slots=n_slots, max_tasks=max_tasks, n_high=max(1, n_high), seed=seed)
        self.energy = float(data.get("energy", 0.8))
        self.tasks = [{"priority": 1 if is_high(t) else 0, "done": False,
                       "title": t.get("title", "Tâche")} for t in tasks_in]
        while len(self.tasks) < max_tasks:                       # pad to fixed size
            self.tasks.append({"priority": 0, "done": False, "title": None})
        hd = data.get("habits_done", {}) or {}
        self.habit_done = {h: bool(hd.get(h, False)) for h in HABITS}
        return self

    @property
    def obs_dim(self) -> int:
        # slot one-hot + energy + 3 habit flags + per-task (done, priority)
        return self.n_slots + 1 + len(HABITS) + self.max_tasks * 2

    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.energy = 0.8
        self.habit_done = {h: False for h in HABITS}
        # task list: first n_high are high priority, rest low; all pending
        self.tasks = ([{"priority": 1, "done": False} for _ in range(self.n_high)]
                      + [{"priority": 0, "done": False}
                         for _ in range(self.max_tasks - self.n_high)])
        self.goal_key = list(GOALS.keys())[int(self.rng.integers(0, len(GOALS)))]
        return self._obs()

    def _obs(self) -> np.ndarray:
        v = np.zeros(self.obs_dim, dtype=np.float32)
        if self.t < self.n_slots:
            v[self.t] = 1.0
        off = self.n_slots
        v[off] = self.energy
        off += 1
        for i, h in enumerate(HABITS):
            v[off + i] = 1.0 if self.habit_done[h] else 0.0
        off += len(HABITS)
        for i, task in enumerate(self.tasks):
            v[off + 2 * i] = 1.0 if task["done"] else 0.0
            v[off + 2 * i + 1] = float(task["priority"])
        return v

    def _pending(self, priority=None):
        for task in self.tasks:
            if not task["done"] and (priority is None or task["priority"] == priority):
                return task
        return None

    def step(self, action: int):
        act = ACTIVITIES[int(action)]
        if act == "deep_work":
            task = self._pending(priority=1) or self._pending()
            if task is not None and self.energy >= 0.3:
                task["done"] = True
                self.energy -= 0.25
            else:
                self.energy -= 0.1
        elif act == "light_work":
            task = self._pending(priority=0) or self._pending()
            if task is not None and self.energy >= 0.15:
                task["done"] = True
                self.energy -= 0.12
            else:
                self.energy -= 0.08
        elif act in ("exercise", "study", "social"):
            self.habit_done[act] = True
            self.energy -= 0.1 if act != "social" else 0.05
        elif act == "rest":
            self.energy += 0.25

        self.energy = float(np.clip(self.energy, 0.0, 1.0))
        self.t += 1
        done = self.t >= self.n_slots
        reward = self._reward() if done else 0.0
        info = {"reached": self._achieved()} if done else {"reached": False}
        return self._obs(), float(reward), bool(done), info

    # --- goal / reward (depend on the hidden daily intention) --- #
    def _high_done(self):
        return sum(1 for t in self.tasks if t["priority"] == 1 and t["done"])

    def _reward(self):
        if self.goal_key == "productive":
            return self._high_done() / max(1, self.n_high)
        if self.goal_key == "balanced":
            return sum(self.habit_done.values()) / len(HABITS)
        if self.goal_key == "recovery":
            return self.energy
        return 0.0

    def _achieved(self):
        if self.goal_key == "productive":
            return self._high_done() >= self.n_high
        if self.goal_key == "balanced":
            return all(self.habit_done.values())
        if self.goal_key == "recovery":
            return self.energy >= 0.6
        return False

    # --- language interface (shared vocab between state and goal targets) --- #
    def text_state(self) -> str:
        d = {h: ("done" if self.habit_done[h] else "todo") for h in HABITS}
        return (f"Day. Energy {_bucket(self.energy)}. Exercise {d['exercise']}. "
                f"Study {d['study']}. Social {d['social']}. "
                f"High tasks done {self._high_done()} of {self.n_high}.")

    def goal_instruction(self) -> str:
        return GOALS[self.goal_key][0]

    def true_goal_key(self):
        return self.goal_key

    def goal_menu(self):
        return [(k, v[1]) for k, v in GOALS.items()]

    def goal_state_text(self, key=None) -> str:
        """Canonical ideal end-of-day state for a goal, in the SAME vocab as
        text_state() so the bridge maps it in-distribution."""
        key = key or self.goal_key
        if key == "productive":
            return (f"Day. Energy low. Exercise todo. Study todo. Social todo. "
                    f"High tasks done {self.n_high} of {self.n_high}.")
        if key == "balanced":
            return (f"Day. Energy medium. Exercise done. Study done. Social done. "
                    f"High tasks done 0 of {self.n_high}.")
        if key == "recovery":
            return (f"Day. Energy high. Exercise todo. Study todo. Social todo. "
                    f"High tasks done 0 of {self.n_high}.")
        return self.text_state()
