"""Day-planning service — single source of truth for the web app AND the CLI.

Loads the trained world model once, builds a day (from your real data if
mydata/today.yaml exists, else synthetic), turns an intention into a goal, and
returns the plan the world model proposes via the latent bridge.
"""
from __future__ import annotations

import os
import random

import yaml

from . import checkpoint
from .controller import MPCController
from .env.lifeworld import LifeWorld, ACTIVITIES, HABITS, GOALS
from .intent import classify_intention
from .llm import Planner
from .userdata import load_today

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("LIFE_CKPT", os.path.join(ROOT, "runs", "life_smoke", "model.pt"))

ACTIVITY_LABELS = {
    "deep_work": "Travail concentré", "light_work": "Travail léger",
    "exercise": "Sport", "study": "Étude", "social": "Social", "rest": "Repos",
}

_CACHE = {}


def load_model(device="cpu"):
    if "m" in _CACHE:
        return _CACHE["m"], _CACHE["cfg"]
    if os.path.exists(CKPT):
        m, cfg, _, _ = checkpoint.load(CKPT, device)
    else:  # no checkpoint yet -> train a quick one and save it
        from .train import train
        with open(os.path.join(ROOT, "configs", "life.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        m = train(cfg, verbose=False)
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        checkpoint.save(CKPT, m, cfg, m["obs_dim"], m["n_actions"])
    _CACHE["m"], _CACHE["cfg"] = m, cfg
    return m, cfg


def build_env(cfg, data=None):
    data = data if data is not None else load_today()
    ns, mt = cfg["env"].get("n_slots", 10), cfg["env"].get("max_tasks", 4)
    if data:
        return LifeWorld.from_data(data, n_slots=ns, max_tasks=mt), True
    return LifeWorld(n_slots=ns, max_tasks=mt, n_high=cfg["env"].get("n_high", 2),
                     seed=random.randint(0, 10**6)), False


def plan_day(intention: str, device="cpu", data=None) -> dict:
    m, cfg = load_model(device)
    env, real = build_env(cfg, data)
    goal_key = classify_intention(intention)
    env.goal_key = goal_key  # the intention overrides any sampled goal

    ctrl = MPCController(m["encoder"], m["world_model"], m["bridge"], m["featurizer"],
                         Planner(backend="mock"),
                         horizon=cfg["control"]["horizon"], n_samples=cfg["control"]["n_samples"],
                         gamma=cfg["control"]["gamma"], guide_weight=cfg["control"]["guide_weight"],
                         device=device)

    obs = env._obs()
    timeline, done, info, reward = [], False, {"reached": False}, 0.0
    while not done:
        slot = env.t
        before = {i for i, t in enumerate(env.tasks) if t["done"]}
        a = ctrl.act(obs, env=env, guided=True)
        name = ACTIVITIES[a]
        obs, reward, done, info = env.step(a)
        newly = {i for i, t in enumerate(env.tasks) if t["done"]} - before
        completed = [env.tasks[i].get("title") for i in newly if env.tasks[i].get("title")]
        timeline.append({
            "slot": slot + 1, "activity": name, "label": ACTIVITY_LABELS.get(name, name),
            "energy": round(env.energy, 2),
            "habits_done": [h for h in HABITS if env.habit_done[h]],
            "completed": completed,
        })

    return {
        "goal_key": goal_key, "goal_intention": GOALS[goal_key][0], "goal_desc": GOALS[goal_key][1],
        "achieved": bool(info.get("reached")), "score": round(float(reward), 2),
        "final_energy": round(env.energy, 2), "high_done": env._high_done(), "n_high": env.n_high,
        "real_data": real, "habits_done": [h for h in HABITS if env.habit_done[h]],
        "timeline": timeline,
        "tasks": [{"title": t.get("title"),
                   "priority": "haute" if t["priority"] == 1 else "basse", "done": t["done"]}
                  for t in env.tasks if t.get("title")],
    }
