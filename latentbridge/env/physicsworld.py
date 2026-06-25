"""PhysicsWorld — continuous physics, now N-D + gravity + soft-landing + VISION.

Shellia's body-prep ground. Configurable along three axes of difficulty:
  - dims: 2 or 3 (étape 4: a 3-D lander).
  - gravity + soft_landing: must DECELERATE and arrive slowly (real control).
  - obs_mode "pixels" (étape 5): she no longer reads her coordinates — she gets a
    small IMAGE (2 stacked frames so velocity is inferable from motion) and must
    learn to SEE. This is the leap from "knowing the state" to "perceiving" it.

Same interface as the other worlds, so the whole LatentBridge pipeline reuses it.
"""
from __future__ import annotations

import numpy as np

VSCALE = 2.0


def _axis_actions(dims: int):
    acts = [np.zeros(dims, dtype=float)]            # 0 = idle
    for i in range(dims):                           # then +axis / -axis each
        e = np.zeros(dims); e[i] = 1.0; acts.append(e.copy())
        e = np.zeros(dims); e[i] = -1.0; acts.append(e.copy())
    return acts                                     # length 2*dims + 1


def _b(v):
    return int(round(float(np.clip(v, 0, 1)) * 10))


class PhysicsWorld:
    def __init__(self, dims: int = 2, dt: float = 0.1, thrust: float = 2.0,
                 drag: float = 0.10, gravity: float = 0.0, max_steps: int = 60,
                 reach_radius: float = 0.12, soft_landing: bool = False,
                 land_speed: float = 0.4, obs_mode: str = "state", img_size: int = 16,
                 seed: int = 0):
        self.dims = int(dims)
        self.dt, self.thrust, self.drag, self.gravity = dt, thrust, drag, gravity
        self.max_steps, self.reach_radius = int(max_steps), reach_radius
        self.soft_landing, self.land_speed = bool(soft_landing), land_speed
        self.obs_mode, self.img_size = obs_mode, int(img_size)
        if obs_mode == "pixels" and self.dims != 2:
            raise ValueError("pixels obs_mode is 2-D only")
        self._acts = _axis_actions(self.dims)
        self.action_space = len(self._acts)
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def obs_dim(self) -> int:
        if self.obs_mode == "pixels":
            return 2 * 2 * self.img_size * self.img_size   # 2 frames x 2 channels
        return 3 * self.dims                                # pos + vel + target

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = self.rng.uniform(0.15, 0.85, size=self.dims)
        self.vel = np.zeros(self.dims, dtype=float)
        self.target = self.rng.uniform(0.15, 0.85, size=self.dims)
        self.t = 0
        self._prev_frame = self._frame()
        return self._obs()

    def _frame(self) -> np.ndarray:
        S = self.img_size
        img = np.zeros((2, S, S), dtype=np.float32)

        def stamp(ch, p):
            cx, cy = int(np.clip(p[0], 0, 1) * (S - 1)), int(np.clip(p[1], 0, 1) * (S - 1))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < S and 0 <= y < S:
                        img[ch, x, y] = 1.0
        stamp(0, self.pos)
        stamp(1, self.target)
        return img

    def _obs(self) -> np.ndarray:
        if self.obs_mode == "pixels":
            cur = self._frame()
            return np.concatenate([self._prev_frame.reshape(-1), cur.reshape(-1)]).astype(np.float32)
        return np.concatenate([self.pos, self.vel / VSCALE, self.target]).astype(np.float32)

    def step(self, action: int):
        f = self._acts[int(action)]
        self.vel += f * self.thrust * self.dt
        self.vel[-1] -= self.gravity * self.dt          # gravity on the last axis
        self.vel *= (1.0 - self.drag)
        self._prev_frame = self._frame()                # pre-move frame -> prev
        self.pos += self.vel * self.dt
        for i in range(self.dims):                       # walls: clamp + bounce
            if self.pos[i] < 0.0:
                self.pos[i] = 0.0; self.vel[i] *= -0.5
            elif self.pos[i] > 1.0:
                self.pos[i] = 1.0; self.vel[i] *= -0.5
        self.t += 1
        dist = float(np.linalg.norm(self.pos - self.target))
        speed = float(np.linalg.norm(self.vel))
        reached = dist < self.reach_radius and (not self.soft_landing or speed < self.land_speed)
        # dense distance reward; when soft-landing, also nudge to slow down near target
        reward = -dist + (1.0 if reached else 0.0)
        if self.soft_landing and dist < 2 * self.reach_radius:
            reward -= 0.3 * speed
        done = reached or self.t >= self.max_steps
        return self._obs(), float(reward), bool(done), {"reached": reached}

    # --- language interface (bucketed, dims-aware) --- #
    def _coords(self, p):
        names = ["x", "y", "z"]
        return " ".join(f"{names[i]}{_b(p[i])}" for i in range(self.dims))

    def text_state(self) -> str:
        sp = "lent" if np.linalg.norm(self.vel) < self.land_speed else "rapide"
        return f"Objet en {self._coords(self.pos)}, vitesse {sp}. Cible {self._coords(self.target)}."

    def true_goal_key(self):
        return ("target",) + tuple(_b(self.target[i]) for i in range(self.dims))

    def goal_state_text(self, key=None) -> str:
        if key is None:
            key = self.true_goal_key()
        b = key[1:]
        names = ["x", "y", "z"]
        coords = " ".join(f"{names[i]}{b[i]}" for i in range(self.dims))
        return f"Objet en {coords}, vitesse lent. Cible {coords}."
