"""Load the user's real day (tasks / habits / energy) from mydata/today.yaml.

This is the bridge from the synthetic world to *your actual life*: the trained
world model stays the same, we just initialize the day state from real data.
"""
from __future__ import annotations

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = os.path.join(ROOT, "mydata", "today.yaml")


def load_today(path: str | None = None):
    """Return the parsed day dict, or None if no file (falls back to synthetic)."""
    p = path or TODAY
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if data.get("tasks") or data.get("habits_done") or "energy" in data else None
