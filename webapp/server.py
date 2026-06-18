"""Web backend for the personal assistant.

Thin layer over latentbridge.planner_service (the same engine the Hermes CLI
uses). Serves:
  GET  /api/state  -> today's structure (your real tasks/habits/energy if
                      mydata/today.yaml exists, else a synthetic day)
  POST /api/plan   -> given a free-text intention, the day plan the world model
                      proposes (LLM intention -> bridge -> plan)
  GET  /api/lab    -> research leaderboard from loop/results.jsonl

Run:  python webapp/server.py   (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import json
import os
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

from latentbridge.env.lifeworld import HABITS, GOALS
from latentbridge import planner_service as svc
from latentbridge.userdata import load_today

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Personal AI — LatentBridge")


class PlanReq(BaseModel):
    intention: str = ""


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/state")
def api_state():
    _, cfg = svc.load_model()
    env, real = svc.build_env(cfg)
    tasks = [{"title": t.get("title"),
              "priority": "haute" if t["priority"] == 1 else "basse",
              "done": t["done"]} for t in env.tasks if t.get("title")]
    return {
        "n_slots": env.n_slots,
        "energy": round(env.energy, 2),
        "real_data": real,
        "habits": HABITS,
        "habits_done": [h for h in HABITS if env.habit_done[h]],
        "tasks": tasks,
        "goals": [{"key": k, "intention": v[0], "desc": v[1]} for k, v in GOALS.items()],
    }


@app.post("/api/plan")
def api_plan(req: PlanReq):
    return svc.plan_day(req.intention)


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
            "n_bridge": len(bridge), "n_control": len(control),
        },
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
