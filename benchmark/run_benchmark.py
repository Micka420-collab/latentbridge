"""LatentBridge benchmark — reproducible, honest, one command.

Measures the central claim across every environment: does coupling a FROZEN LLM
to a TRAINED world model through the learned LATENT BRIDGE actually help control?
The signal is `guidance_gain` = success WITH the bridge minus success WITHOUT it,
on held-out seeds. A `--ablation` pass turns the bridge fully off (align/ground/
cycle = 0, guide_weight = 0) as the scientific control.

    python benchmark/run_benchmark.py                 # core envs
    python benchmark/run_benchmark.py --ablation      # + no-bridge control
    python benchmark/run_benchmark.py --only desktop,calc

Writes benchmark/results.json and benchmark/RESULTS.md (real numbers, CPU).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENVS = [
    ("gridworld",    "configs/base.yaml"),
    ("lifeworld",    "configs/life.yaml"),
    ("physics2d",    "configs/physics.yaml"),
    ("physics3d",    "configs/physics3d.yaml"),
    ("desktopworld", "configs/desktop.yaml"),
    ("simcalc",      "configs/calc.yaml"),
]
ABLATE = {"loss.align": 0.0, "loss.ground": 0.0, "loss.cycle": 0.0, "control.guide_weight": 0.0}


def run(config, tag, overrides=None):
    cmd = [sys.executable, "-m", "latentbridge.experiment", "--config", config, "--tag", tag]
    for k, v in (overrides or {}).items():
        cmd += ["--set", f"{k}={v}"]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    with open(os.path.join(ROOT, "runs", tag, "metrics.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true", help="also run the no-bridge control")
    ap.add_argument("--only", default=None, help="comma list of env names")
    args = ap.parse_args()
    envs = ENVS if not args.only else [e for e in ENVS if e[0] in args.only.split(",")]

    rows = []
    for name, cfg in envs:
        print(f"[bench] {name} …", flush=True)
        m = run(cfg, f"bench_{name}")
        row = {
            "env": name, "config": cfg,
            "score": round(m["score"], 3),
            "success_guided": round(m["success_rate_guided"], 3),
            "success_unguided": round(m["success_rate_unguided"], 3),
            "guidance_gain": round(m["guidance_gain"], 3),
            "align_cosine": round(m["align_cosine_heldout"], 4),
            "wall_sec": m.get("wall_time_sec"),
        }
        if args.ablation:
            a = run(cfg, f"bench_{name}_ablate", ABLATE)
            row["ablation_success"] = round(a["success_rate_guided"], 3)
        rows.append(row)
        print(f"        gain={row['guidance_gain']:+.2f}  guided={row['success_guided']:.2f}", flush=True)

    out = {"rows": rows, "device": "cpu", "note": "held-out eval; higher is better"}
    with open(os.path.join(ROOT, "benchmark", "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # markdown table
    cols = ["env", "score", "success_guided", "success_unguided", "guidance_gain", "align_cosine"]
    if args.ablation:
        cols.insert(4, "ablation_success")
    lines = ["# LatentBridge — résultats du benchmark", "",
             "Held-out, CPU. `guidance_gain` = succès avec le pont − sans le pont (le cœur de la preuve).",
             "" , "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines += ["", "Reproduire : `python benchmark/run_benchmark.py --ablation`"]
    with open(os.path.join(ROOT, "benchmark", "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
