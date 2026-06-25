"""Measure a LatentBridge variant and give a clear verdict vs baseline.

    python scripts/measure.py --config configs/life.yaml --set loss.align=0.5
    python scripts/measure.py --config configs/base.yaml --set model.latent_dim=128 --baseline 0.57

Runs the real held-out experiment, prints the score, and says BETTER / WORSE so
an improving agent keeps only what actually helps. Never edits anything.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASELINE = {"life": 0.88, "base": 0.57}  # see AGENTS.md / loop/results.jsonl


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/life.yaml")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--baseline", type=float, default=None)
    args = ap.parse_args()

    tag = "measure_" + str(int(time.time()))
    cmd = [sys.executable, "-m", "latentbridge.experiment",
           "--config", args.config, "--tag", tag]
    for ov in args.overrides:
        cmd += ["--set", ov]
    subprocess.run(cmd, cwd=ROOT, check=True)

    with open(os.path.join(ROOT, "runs", tag, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    score = m["score"]

    base = args.baseline
    if base is None:
        key = "life" if "life" in args.config else "base"
        base = DEFAULT_BASELINE.get(key)

    print("\n" + "=" * 48)
    print(f"score          = {score:.4f}")
    print(f"guidance_gain  = {m.get('guidance_gain')}")
    print(f"align_cosine   = {m.get('align_cosine_heldout')}")
    if base is not None:
        delta = score - base
        verdict = "BETTER (keep)" if delta > 0 else "WORSE (discard)"
        print(f"baseline       = {base:.4f}  ->  delta {delta:+.4f}  => {verdict}")
    print("=" * 48)


if __name__ == "__main__":
    main()
