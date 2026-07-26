"""Paraphrase-robustness probe — can the goal be phrased freely?

The judge (evaluate.py) always passes the env's CANONICAL goal text through the
bridge. A real frozen LLM will not: it will say "go to row 2, column 3" in a
hundred different ways. This probe measures guided success when the planner
emits PARAPHRASES of the goal instead of the canonical text — the capability
the language featurizer must support for the frozen-LLM story to hold.

Never touches evaluate.py / env. Gridworld-focused (cell goals).

    python scripts/paraphrase_probe.py --config configs/base.yaml
    python scripts/paraphrase_probe.py --config configs/base.yaml \
        --set model.text_features=stransformer --seeds 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml                                              # noqa: E402

from latentbridge.controller import MPCController        # noqa: E402
from latentbridge.env import make_env                    # noqa: E402
from latentbridge.train import train                     # noqa: E402


TEMPLATES = {
    "canonical":  None,   # env.goal_state_text — the judge's regime (control)
    "polite":     lambda env, r, c: f"The agent should end up at row {r}, column {c}.",
    "imperative": lambda env, r, c: f"Please move until you are standing at row {r} col {c}.",
    "swapped":    lambda env, r, c: f"Target position: column {c}, row {r}.",
    "verbose":    lambda env, r, c: f"Go to the cell in row {r} and column {c} of the grid.",
    "terse":      lambda env, r, c: f"row {r} col {c}",
}


class ParaphrasingPlanner:
    """Mock planner that reveals the true goal but phrases it via `template`."""

    def __init__(self, template):
        self.template = template

    def propose(self, env, intention=None):
        key = env.true_goal_key()
        if self.template is None:
            return {"subgoal_text": env.goal_state_text(key), "goal_key": key}
        _, r, c = key
        return {"subgoal_text": self.template(env, r, c), "goal_key": key}


def run_guided(modules, cfg, planner, seeds):
    ctrl = MPCController(modules["encoder"], modules["world_model"],
                         modules["bridge"], modules["featurizer"], planner,
                         horizon=cfg["control"]["horizon"],
                         n_samples=cfg["control"]["n_samples"],
                         gamma=cfg["control"]["gamma"],
                         guide_weight=cfg["control"]["guide_weight"],
                         optimizer=cfg["control"].get("optimizer", "shooting"),
                         revisit_weight=cfg["control"].get("revisit_weight", 0.0),
                         revisit_mem=cfg["control"].get("revisit_mem", 6),
                         seed=cfg["seed"])
    succ = []
    for s in seeds:
        env = make_env(cfg, seed=s)
        obs = env.reset(seed=s)
        done, info = False, {}
        while not done:
            obs, _, done, info = env.step(ctrl.act(obs, env=env, guided=True))
        succ.append(1.0 if info.get("reached") else 0.0)
    return float(np.mean(succ))


def _set_in(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    d[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1, help="training seeds to average")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in args.overrides:
        key, _, val = ov.partition("=")
        _set_in(cfg, key.strip(), val.strip())
    if cfg["env"].get("kind", "gridworld") != "gridworld":
        raise SystemExit("probe supports gridworld cell goals only")

    backend = cfg["model"].get("text_features", "hashing")
    results = {name: [] for name in TEMPLATES}
    for ts in range(args.seeds):
        c = dict(cfg); c["seed"] = cfg["seed"] + ts
        m = train(c, verbose=False)
        eval_seeds = list(range(c["seed"] + 10_000, c["seed"] + 10_000 + args.episodes))
        for name, tpl in TEMPLATES.items():
            sr = run_guided(m, c, ParaphrasingPlanner(tpl), eval_seeds)
            results[name].append(sr)
            print(f"  seed {c['seed']} | {name:11s} success={sr:.3f}", flush=True)

    print("\n" + "=" * 52)
    print(f"PARAPHRASE PROBE  backend={backend}  "
          f"({args.seeds} seed(s) x {args.episodes} held-out episodes)")
    for name in TEMPLATES:
        v = results[name]
        print(f"{name:11s} success = {np.mean(v):.3f} ± {np.std(v):.3f}")
    canon = np.mean(results["canonical"])
    para = np.mean([np.mean(results[n]) for n in TEMPLATES if n != "canonical"])
    print(f"\nparaphrase retention = {para:.3f} / {canon:.3f} "
          f"({100 * para / max(canon, 1e-9):.0f}% of canonical)")
    print("=" * 52)


if __name__ == "__main__":
    main()
