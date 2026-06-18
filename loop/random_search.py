"""Built-in continuous architecture search — works TODAY, no external repo needed.

It does exactly what Arbor will do, just simpler: sample a point in the search
space, run one experiment as a subprocess, read metrics.json, track the best,
append to results.jsonl, repeat. Arbor/Hermes are a heavier upgrade of this same
contract (see ../integrations/).

    python loop/random_search.py --trials 30
    python loop/random_search.py --trials 0     # run forever (Ctrl-C to stop)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "loop", "results.jsonl")


def sample(knobs, rng):
    overrides = {}
    for key, spec in knobs.items():
        if spec["type"] == "float":
            overrides[key] = round(float(rng.uniform(spec["low"], spec["high"])), 5)
        elif spec["type"] == "choice":
            overrides[key] = spec["values"][int(rng.integers(0, len(spec["values"])))]
    return overrides


def run_trial(overrides, tag):
    cmd = [sys.executable, "-m", "latentbridge.experiment",
           "--config", os.path.join("configs", "base.yaml"), "--tag", tag]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    with open(os.path.join(ROOT, "runs", tag, "metrics.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20, help="0 = run forever")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "loop", "search_space.yaml")) as f:
        space = yaml.safe_load(f)
    knobs = space["knobs"]
    rng = np.random.default_rng(args.seed)

    best = {"score": -1e9}
    i = 0
    print("=== LatentBridge continuous search ===  (Ctrl-C to stop)\n")
    while args.trials == 0 or i < args.trials:
        # every 5th trial: run the no-bridge control group for comparison
        if i % 5 == 4 and space.get("controls"):
            ov = dict(space["controls"][0]["set"])
            tag = f"trial_{i:04d}_control"
        else:
            ov = sample(knobs, rng)
            tag = f"trial_{i:04d}"
        try:
            metrics = run_trial(ov, tag)
        except subprocess.CalledProcessError as e:
            print(f"[trial {i}] FAILED: {e}")
            i += 1
            continue

        rec = {"trial": i, "tag": tag, "overrides": ov, "metrics": metrics}
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")

        score = metrics.get("score", -1e9)
        flag = ""
        if score > best["score"]:
            best = {"score": score, "tag": tag, "overrides": ov, "metrics": metrics}
            flag = "  <-- NEW BEST"
        print(f"[trial {i:04d}] score={score:.4f} "
              f"sr_g={metrics.get('success_rate_guided'):.2f} "
              f"gain={metrics.get('guidance_gain'):+.2f} "
              f"align={metrics.get('align_cosine_heldout'):.3f}{flag}")
        i += 1

    print("\n=== BEST ===")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
