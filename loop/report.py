"""Summarize loop/results.jsonl: leaderboard + bridge-vs-control comparison.

    python loop/report.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "loop", "results.jsonl")


def main():
    if not os.path.exists(RESULTS):
        print("no results yet — run: python loop/random_search.py --trials 20")
        return
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    if not rows:
        print("results file is empty")
        return

    def m(r, k):
        return r["metrics"].get(k)

    rows.sort(key=lambda r: m(r, "score"), reverse=True)

    print(f"\n{'tag':22s} {'score':>8s} {'sr_guid':>8s} {'sr_base':>8s} {'gain':>7s} {'align':>7s}")
    print("-" * 70)
    for r in rows[:15]:
        print(f"{r['tag']:22s} {m(r,'score'):8.3f} {m(r,'success_rate_guided'):8.2f} "
              f"{m(r,'success_rate_unguided'):8.2f} {m(r,'guidance_gain'):+7.2f} "
              f"{m(r,'align_cosine_heldout'):7.3f}")

    bridge = [r for r in rows if "control" not in r["tag"]]
    control = [r for r in rows if "control" in r["tag"]]

    def avg(rs, k):
        vals = [m(r, k) for r in rs if m(r, k) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n=== Bridge ON vs control (bridge OFF) ===")
    print(f"  bridge ON  ({len(bridge):2d} runs): mean guidance_gain = {avg(bridge,'guidance_gain'):+.3f} "
          f"| mean align = {avg(bridge,'align_cosine_heldout'):.3f}")
    print(f"  control    ({len(control):2d} runs): mean guidance_gain = {avg(control,'guidance_gain'):+.3f} "
          f"| mean align = {avg(control,'align_cosine_heldout'):.3f}")

    best = rows[0]
    print("\n=== BEST CONFIG ===")
    print(json.dumps({"tag": best["tag"], "overrides": best["overrides"],
                      "score": m(best, "score")}, indent=2))


if __name__ == "__main__":
    main()
