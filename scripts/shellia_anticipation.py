"""Shellia-side reader: turn the live desktop anticipation into one sentence.

The Windows side (scripts/anticipate_report.py) perceives Micka's real desktop and
pushes runs/app_world_model/live_anticipation.json here. Shellia's brain (the hermes
agent, which has a terminal tool) runs THIS to know what Micka is doing / about to do:

    python3 scripts/shellia_anticipation.py

Pure stdlib so it runs on the VM with no deps. Reports staleness honestly.
"""
import json
import os
import sys
from datetime import datetime, timezone

CANDIDATES = [
    os.environ.get("SHELLIA_ANTICIPATION_FILE", ""),
    "/root/latentbridge/live_anticipation.json",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "runs", "app_world_model", "live_anticipation.json"),
]


def load():
    for p in CANDIDATES:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f), p
    return None, None


def age_str(ts):
    try:
        t = datetime.fromisoformat(ts)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        secs = (datetime.now(timezone.utc) - t).total_seconds()
        if secs < 90:
            return f"il y a {int(secs)}s (frais)"
        if secs < 3600:
            return f"il y a {int(secs/60)}min"
        return f"il y a {int(secs/3600)}h (perime)"
    except Exception:
        return "age inconnu"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rec, path = load()
    if rec is None:
        print("(pas d'anticipation: le rapport Windows n'a pas encore pousse de fichier)")
        return 1
    now = rec.get("now") or "?"
    nxt = rec.get("anticipated") or "?"
    win = (rec.get("window") or "?")[:50]
    grow = ", ".join(rec.get("grow") or []) or "(stable)"
    print(f"Micka — fenetre active: {win}")
    print(f"  maintenant      : {now}")
    print(f"  va probablement : {nxt}   [a venir: {grow}]")
    print(f"  source: {os.path.basename(path)} — {age_str(rec.get('ts',''))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
