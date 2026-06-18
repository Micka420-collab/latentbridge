"""CLI the Hermes assistant calls to plan your day from an intention.

    python -m latentbridge.plan_cli "je suis crevé, juste me reposer"
    python -m latentbridge.plan_cli --json "finir mes tâches importantes"

Reads your real day from mydata/today.yaml when present. Output is WhatsApp-
friendly markdown so Hermes can relay it verbatim.
"""
from __future__ import annotations

import argparse
import json
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .planner_service import plan_day

EMOJI = {"deep_work": "🎯", "light_work": "✍️", "exercise": "🏃",
         "study": "📚", "social": "💬", "rest": "😌"}
GOAL_TITLE = {"productive": "Journée productive", "balanced": "Journée équilibrée",
              "recovery": "Récupération / repos"}


def to_markdown(p: dict) -> str:
    lines = [f"🗓️ *Plan du jour* — _{GOAL_TITLE.get(p['goal_key'], p['goal_key'])}_",
             f"Source : {'tes vraies données' if p['real_data'] else 'journée type'} · "
             f"énergie départ {int(p['timeline'][0]['energy'] * 100)}%", ""]
    for s in p["timeline"]:
        extra = ""
        if s["completed"]:
            extra = " — ✅ " + ", ".join(f'"{c}"' for c in s["completed"])
        elif s["habits_done"]:
            extra = " · habitude ✓"
        lines.append(f"{s['slot']:>2}. {EMOJI.get(s['activity'],'•')} {s['label']}{extra}")
    status = "✅ objectif atteint" if p["achieved"] else "⚠️ objectif partiel"
    lines += ["", f"*Bilan* : {status} · score {int(p['score']*100)}% · "
              f"énergie finale {int(p['final_energy']*100)}% · "
              f"{p['high_done']}/{p['n_high']} tâches importantes"]
    return "\n".join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("intention", nargs="*", help="your intention for today")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    intention = " ".join(args.intention) or "journée équilibrée"

    p = plan_day(intention)
    if args.json:
        print(json.dumps(p, ensure_ascii=False, indent=2))
    else:
        print(to_markdown(p))


if __name__ == "__main__":
    main()
