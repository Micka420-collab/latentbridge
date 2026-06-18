"""Map a free-text intention to a day goal (shared by the web app and the CLI)."""
from __future__ import annotations

import os

from .env.lifeworld import LifeWorld, GOALS
from .llm import Planner


def classify_intention(text: str) -> str:
    # Prefer the real LLM (it reads nuance); keyword fallback keeps it offline.
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            key = Planner(backend="openrouter")._classify_llm(LifeWorld(seed=0), intention=text)
            if key in GOALS:
                return key
        except Exception:
            pass
    t = (text or "").lower()
    if any(w in t for w in ["repos", "rest", "fatig", "tired", "exhaust", "recharg",
                            "relax", "calme", "souffl", "épuis", "epuis", "dormir"]):
        return "recovery"
    if any(w in t for w in ["travail", "work", "task", "tâche", "tache", "deadline",
                            "fini", "finir", "important", "product", "bosser", "projet",
                            "rapport", "présentation", "presentation"]):
        return "productive"
    return "balanced"
