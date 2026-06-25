"""Single-experiment entry point — the contract Arbor (or you) calls.

    python -m latentbridge.experiment --config configs/base.yaml
    python -m latentbridge.experiment --config configs/base.yaml --set loss.align=0.5 --set model.latent_dim=96

It trains, evaluates on held-out seeds, and writes runs/<tag>/metrics.json plus a
snapshot of the exact config used. The primary metric is metrics["score"]
(higher is better). Arbor mutates config keys, runs this, reads metrics.json,
keeps winners, prunes losers.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .train import train
from .evaluate import evaluate
from . import checkpoint


def _set_in(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _cast(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    help="dotted.key=value override (repeatable)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for ov in args.overrides:
        key, _, val = ov.partition("=")
        _set_in(cfg, key.strip(), _cast(val.strip()))

    tag = args.tag or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    modules = train(cfg, device=args.device)
    metrics = evaluate(modules, cfg, device=args.device)
    metrics["wall_time_sec"] = round(time.time() - t0, 2)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    # the learned "self" — portable to any new body (env / robot)
    checkpoint.save(os.path.join(out_dir, "model.pt"), modules, cfg,
                    modules["obs_dim"], modules["n_actions"])

    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nprimary score = {metrics['score']:.4f}  ->  {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
