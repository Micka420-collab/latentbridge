"""Web backend for the personal assistant.

Loads the trained "self" (a LifeWorld checkpoint) and serves:
  GET  /api/state  -> today's structure (tasks, habits, energy, goals menu)
  POST /api/plan   -> given a free-text intention, returns the day plan the
                      world model proposes (LLM intention -> bridge -> plan)
  GET  /api/lab    -> research leaderboard from loop/results.jsonl

Run:  python webapp/server.py   (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from latentbridge import checkpoint
from latentbridge.controller import MPCController
from latentbridge.env.lifeworld import LifeWorld, ACTIVITIES, HABITS, GOALS
from latentbridge.llm import Planner

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
CKPT = os.environ.get("LIFE_CKPT", os.path.join(ROOT, "runs", "life_smoke", "model.pt"))

ACTIVITY_LABELS = {
    "deep_work": "Travail concentré", "light_work": "Travail léger",
    "exercise": "Sport", "study": "Étude", "social": "Social", "rest": "Repos",
}

app = FastAPI(title="Personal AI — LatentBridge")
STATE = {"modules": None, "cfg": None}


def _ensure_model():
    if STATE["modules"] is not None:
        return
    if not os.path.exists(CKPT):
        # train a quick one if no checkpoint exists yet
        import yaml
        from latentbridge.train import train
        with open(os.path.join(ROOT, "configs", "life.yaml")) as f:
            cfg = yaml.safe_load(f)
        m = train(cfg, verbose=False)
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        checkpoint.save(CKPT, m, cfg, m["obs_dim"], m["n_actions"])
        STATE["modules"], STATE["cfg"] = m, cfg
    else:
        m, cfg, _, _ = checkpoint.load(CKPT)
        STATE["modules"], STATE["cfg"] = m, cfg


def _classify_intention(text: str) -> str:
    """LLM picks the goal that matches the intention; keyword fallback offline."""
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            p = Planner(backend="openrouter")
            env = LifeWorld(seed=random.randint(0, 10**6))
            key = p._classify_llm(env, intention=text)
            if key in GOALS:
                return key
        except Exception:
            pass
    t = (text or "").lower()
    if any(w in t for w in ["repos", "rest", "fatig", "tired", "exhaust", "recharg", "relax", "calme"]):
        return "recovery"
    if any(w in t for w in ["travail", "work", "task", "tâche", "deadline", "fini", "important", "product", "bosser", "projet"]):
        return "productive"
    return "balanced"


class PlanReq(BaseModel):
    intention: str = ""


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/state")
def api_state():
    env = LifeWorld(seed=random.randint(0, 10**6))
    tasks = [{"priority": "haute" if t["priority"] == 1 else "basse",
              "done": t["done"]} for t in env.tasks]
    return {
        "n_slots": env.n_slots,
        "energy": round(env.energy, 2),
        "habits": HABITS,
        "tasks": tasks,
        "goals": [{"key": k, "intention": v[0], "desc": v[1]} for k, v in GOALS.items()],
    }


@app.post("/api/plan")
def api_plan(req: PlanReq):
    _ensure_model()
    m, cfg = STATE["modules"], STATE["cfg"]
    goal_key = _classify_intention(req.intention)

    env = LifeWorld(n_slots=cfg["env"].get("n_slots", 10),
                    max_tasks=cfg["env"].get("max_tasks", 4),
                    n_high=cfg["env"].get("n_high", 2),
                    seed=random.randint(0, 10**6))
    env.goal_key = goal_key  # the user's intention overrides the random one

    ctrl = MPCController(m["encoder"], m["world_model"], m["bridge"], m["featurizer"],
                         Planner(backend="mock"),
                         horizon=cfg["control"]["horizon"],
                         n_samples=cfg["control"]["n_samples"],
                         gamma=cfg["control"]["gamma"],
                         guide_weight=cfg["control"]["guide_weight"])

    obs = env._obs()
    timeline, done, info, reward = [], False, {"reached": False}, 0.0
    while not done:
        slot = env.t
        a = ctrl.act(obs, env=env, guided=True)
        name = ACTIVITIES[a]
        obs, reward, done, info = env.step(a)
        timeline.append({
            "slot": slot + 1,
            "activity": name,
            "label": ACTIVITY_LABELS.get(name, name),
            "energy": round(env.energy, 2),
            "habits_done": [h for h in HABITS if env.habit_done[h]],
            "high_done": env._high_done(),
        })

    return {
        "goal_key": goal_key,
        "goal_intention": GOALS[goal_key][0],
        "goal_desc": GOALS[goal_key][1],
        "achieved": bool(info.get("reached")),
        "score": round(float(reward), 2),
        "final_energy": round(env.energy, 2),
        "habits_done": [h for h in HABITS if env.habit_done[h]],
        "high_done": env._high_done(),
        "n_high": env.n_high,
        "timeline": timeline,
    }


@app.get("/api/lab")
def api_lab():
    path = os.path.join(ROOT, "loop", "results.jsonl")
    if not os.path.exists(path):
        return {"rows": [], "summary": None}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows.sort(key=lambda r: r["metrics"].get("score", -9), reverse=True)
    bridge = [r for r in rows if "control" not in r["tag"]]
    control = [r for r in rows if "control" in r["tag"]]

    def avg(rs, k):
        v = [r["metrics"].get(k) for r in rs if r["metrics"].get(k) is not None]
        return round(sum(v) / len(v), 3) if v else None

    top = [{"tag": r["tag"], "score": round(r["metrics"]["score"], 3),
            "gain": round(r["metrics"].get("guidance_gain", 0), 3),
            "align": round(r["metrics"].get("align_cosine_heldout", 0), 3)}
           for r in rows[:8]]
    return {
        "rows": top,
        "summary": {
            "bridge_gain": avg(bridge, "guidance_gain"),
            "control_gain": avg(control, "guidance_gain"),
            "bridge_align": avg(bridge, "align_cosine_heldout"),
            "control_align": avg(control, "align_cosine_heldout"),
            "n_bridge": len(bridge), "n_control": len(control),
        },
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
