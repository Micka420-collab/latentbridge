"""The comparison harness — this is what turns "it works on our task" into a
defensible "more efficient than X at equal budget".

It runs, in the SAME env and at an EQUAL environment-episode budget, over multiple
seeds:

  random        chance floor
  ppo           model-free RL (actor-critic)
  dqn           model-free RL (value-based)
  llm_only      planner goal, no world model       (isolates the world model)
  world_model   LatentBridge with the bridge ablated (align=0) — the control group
  latentbridge  the full architecture (world model + LLM bridge)

Two regimes (BENCHMARKING.md §2 + §4):
  visible   include_goal_in_obs=True  -> every method can see the goal. A fair
            sample-efficiency contest of the *learning architecture*.
  language  include_goal_in_obs=False -> goal given only in language. Only methods
            with a language channel can succeed; the non-language baselines near
            chance IS the measured value of the bridge.

The equalized axis is environment episodes (sample-efficiency). We also report
wall-time (a compute-efficiency proxy) and parameter count (param-efficiency), so
the table covers the three efficiency axes the doc lists. Output: a markdown table
+ a JSON blob under runs/<tag>/.

    python -m latentbridge.benchmark --config configs/base.yaml \
        --seeds 5 --budget 80 --steps 1500 --regime both --tag grid_v1
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .train import train
from .evaluate import evaluate
from .baselines import train_ppo, train_dqn, RandomPolicy, LLMOnlyPolicy, eval_policy


# --------------------------------------------------------------------------- #
# one (method, seed) run -> {"success_rate", "avg_steps", "params", ...}
# --------------------------------------------------------------------------- #
def _wm_params(m) -> int:
    return int(sum(p.numel() for k in ("encoder", "decoder", "world_model", "bridge")
                   for p in m[k].parameters()))


def run_once(method: str, cfg: dict, seed: int, budget: int, steps: int,
             device: str = "cpu") -> dict:
    c = copy.deepcopy(cfg)
    c["seed"] = seed
    t0 = time.time()

    if method == "latentbridge":
        c["train"]["collect_episodes"] = budget
        c["train"]["steps"] = steps
        m = train(c, device=device, verbose=False)
        metr = evaluate(m, c, device=device)
        out = {"success_rate": metr["success_rate_guided"],
               "avg_steps": metr["avg_steps_guided"], "params": _wm_params(m)}

    elif method == "world_model":
        # the align=0 control group: same world model, bridge ablated in BOTH
        # training (no language shaping of the latent) and control (no guidance).
        c["train"]["collect_episodes"] = budget
        c["train"]["steps"] = steps
        c["loss"] = dict(c["loss"], align=0.0, cycle=0.0, ground=0.0)
        c["control"] = dict(c["control"], guide_weight=0.0)
        m = train(c, device=device, verbose=False)
        metr = evaluate(m, c, device=device)
        out = {"success_rate": metr["success_rate_unguided"],
               "avg_steps": metr["avg_steps_unguided"], "params": _wm_params(m)}

    elif method == "ppo":
        pol, info = train_ppo(c, seed, device=device, episodes_budget=budget)
        metr = eval_policy(pol, c, device=device)
        out = {"success_rate": metr["success_rate"],
               "avg_steps": metr["avg_steps"], "params": info["params"]}

    elif method == "dqn":
        pol, info = train_dqn(c, seed, device=device, episodes_budget=budget)
        metr = eval_policy(pol, c, device=device)
        out = {"success_rate": metr["success_rate"],
               "avg_steps": metr["avg_steps"], "params": info["params"]}

    elif method == "random":
        probe = __import__("latentbridge.env", fromlist=["make_env"]).make_env(c, seed=seed)
        pol = RandomPolicy(probe.action_space, seed=seed)
        metr = eval_policy(pol, c, device=device)
        out = {"success_rate": metr["success_rate"],
               "avg_steps": metr["avg_steps"], "params": 0}

    elif method == "llm_only":
        pol = LLMOnlyPolicy(c, seed=seed)
        metr = eval_policy(pol, c, device=device)
        out = {"success_rate": metr["success_rate"],
               "avg_steps": metr["avg_steps"], "params": 0,
               "degraded": bool(getattr(pol, "degraded", False))}

    else:
        raise ValueError(f"unknown method: {method}")

    out["wall_sec"] = round(time.time() - t0, 2)
    out["budget_episodes"] = budget
    return out


# --------------------------------------------------------------------------- #
# aggregate seeds -> mean / std
# --------------------------------------------------------------------------- #
def aggregate(runs: list[dict]) -> dict:
    sr = np.array([r["success_rate"] for r in runs], dtype=float)
    st = np.array([r["avg_steps"] for r in runs], dtype=float)
    wt = np.array([r["wall_sec"] for r in runs], dtype=float)
    return {
        "success_mean": float(sr.mean()), "success_std": float(sr.std(ddof=0)),
        "steps_mean": float(st.mean()),
        "wall_mean": float(wt.mean()),
        "params": int(runs[0].get("params", 0)),
        "n_seeds": len(runs),
        "degraded": any(r.get("degraded") for r in runs),
        "raw_success": sr.tolist(),
    }


def run_regime(cfg: dict, regime: str, methods: list[str], seeds: list[int],
               budget: int, steps: int, device: str = "cpu") -> dict:
    c = copy.deepcopy(cfg)
    c["env"] = dict(c["env"])
    c["env"]["include_goal_in_obs"] = (regime == "visible")
    results = {}
    for method in methods:
        runs = []
        for sd in seeds:
            r = run_once(method, c, sd, budget, steps, device=device)
            runs.append(r)
            print(f"  [{regime:8s}] {method:12s} seed={sd:<4d} "
                  f"success={r['success_rate']:.3f} steps={r['avg_steps']:.1f} "
                  f"wall={r['wall_sec']:.1f}s", flush=True)
        results[method] = aggregate(runs)
    return results


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
_LABELS = {
    "random": "Random (chance floor)",
    "ppo": "PPO (model-free RL)",
    "dqn": "DQN (model-free RL)",
    "llm_only": "LLM-only (planner, no world model)",
    "world_model": "World model only (bridge ablated, align=0)",
    "latentbridge": "LatentBridge (world model + LLM bridge)",
}
_ORDER = ["random", "ppo", "dqn", "llm_only", "world_model", "latentbridge"]


def render_markdown(meta: dict, regimes: dict) -> str:
    lines = ["# Benchmark — LatentBridge vs reference baselines", ""]
    lines.append(f"- env: `{meta['env_kind']}`  •  budget: **{meta['budget']} episodes** "
                 f"(equalized axis = sample-efficiency)  •  grad steps (WM/LB): {meta['steps']}")
    lines.append(f"- seeds: {meta['n_seeds']}  •  held-out eval episodes: {meta['eval_episodes']} "
                 f"(seeds disjoint: train < +10k, eval = +10k)")
    lines.append(f"- device: {meta['device']}  •  generated for tag `{meta['tag']}`")
    lines.append("")
    for regime, res in regimes.items():
        title = ("Regime A — goal visible in observation (fair learning-efficiency contest)"
                 if regime == "visible" else
                 "Regime B — goal given only in language (measures the bridge's value)")
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Method | Success (mean±std) | Avg steps | Params | Wall (s) |")
        lines.append("|---|---|---|---|---|")
        for m in _ORDER:
            if m not in res:
                continue
            a = res[m]
            label = _LABELS[m] + (" *(degraded→random)*" if a.get("degraded") else "")
            lines.append(f"| {label} | {a['success_mean']:.3f} ± {a['success_std']:.3f} "
                         f"| {a['steps_mean']:.1f} | {a['params']:,} | {a['wall_mean']:.1f} |")
        # the scientific bottom line for this regime
        if "latentbridge" in res and "world_model" in res:
            lb, wm = res["latentbridge"]["success_mean"], res["world_model"]["success_mean"]
            sd = (res["latentbridge"]["success_std"] + res["world_model"]["success_std"])
            delta = lb - wm
            verdict = ("**bridge helps**" if delta > sd and delta > 0
                       else "**not significant at this seed count**")
            lines.append("")
            lines.append(f"> Bridge contribution (LatentBridge − World-model-only): "
                         f"**{delta:+.3f}** success — {verdict}. "
                         f"(Δ vs Σstd={sd:.3f}; scale to ≥5 seeds + a paired t-test to confirm.)")
        lines.append("")
    lines.append("---")
    lines.append("*Honesty note (BENCHMARKING.md §5): this is the built-in task. Until the "
                 "same harness runs on a public benchmark (MiniGrid/BabyAI/Crafter/Atari-100k), "
                 "report \"promising on our task\", not \"state of the art\".*")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0, help="first training seed")
    ap.add_argument("--budget", type=int, default=80, help="env episodes per method")
    ap.add_argument("--steps", type=int, default=1500, help="grad steps for WM/LB")
    ap.add_argument("--regime", choices=["visible", "language", "both"], default="both")
    ap.add_argument("--methods", default=",".join(_ORDER))
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.eval_episodes:
        cfg["eval"]["episodes"] = args.eval_episodes

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [args.seed0 + i for i in range(args.seeds)]
    regimes_to_run = ["visible", "language"] if args.regime == "both" else [args.regime]

    tag = args.tag or time.strftime("bench_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "tag": tag, "env_kind": cfg["env"].get("kind", "gridworld"),
        "budget": args.budget, "steps": args.steps, "n_seeds": args.seeds,
        "eval_episodes": cfg["eval"]["episodes"], "device": args.device,
        "methods": methods,
    }

    print(f"=== benchmark {tag} | env={meta['env_kind']} budget={args.budget} "
          f"steps={args.steps} seeds={args.seeds} regimes={regimes_to_run} ===", flush=True)

    regimes = {}
    t0 = time.time()
    for regime in regimes_to_run:
        print(f"\n--- regime: {regime} ---", flush=True)
        regimes[regime] = run_regime(cfg, regime, methods, seeds,
                                     args.budget, args.steps, device=args.device)
        # checkpoint partial results after each regime
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "regimes": regimes}, f, indent=2)
    meta["total_wall_sec"] = round(time.time() - t0, 1)

    md = render_markdown(meta, regimes)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "regimes": regimes}, f, indent=2)
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + md)
    print(f"\n-> {out_dir}/REPORT.md  ({meta['total_wall_sec']}s total)")


if __name__ == "__main__":
    main()
