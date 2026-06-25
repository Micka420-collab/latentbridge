"""World-model sanity probe — does the learned latent dynamics ACTUALLY model
the world, or is success leaking from elsewhere? (The honest secondary metrics
the SOTA research demanded, never touching the judge evaluate.py / env.)

    python scripts/wm_probe.py --run desktop_baseline

Reports, on HELD-OUT rollouts:
  - 1-step latent error vs a no-op baseline (||z_t - z_{t+1}||). If the model
    error is << the no-op baseline, the dynamics learned real transitions.
  - k-step rollout drift (k=1,3,5): how fast imagination becomes untrustworthy.
    This is the honest stop-signal for how deep MPC planning can go.
All errors are reported relative to the RMS latent norm so they are comparable.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latentbridge import checkpoint                      # noqa: E402
from latentbridge.env import make_env                    # noqa: E402


def collect(cfg, n_episodes, seed_base):
    rng = np.random.default_rng(seed_base)
    eps = []
    for s in range(seed_base, seed_base + n_episodes):
        env = make_env(cfg, seed=s)
        obs = env.reset(seed=s)
        O, A = [obs], []
        done = False
        while not done:
            a = int(rng.integers(0, env.action_space))
            obs, _, done, _ = env.step(a)
            O.append(obs); A.append(a)
        eps.append((np.asarray(O, np.float32), np.asarray(A, np.int64)))
    return eps


@torch.no_grad()
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="desktop_baseline", help="runs/<tag> with model.pt")
    ap.add_argument("--episodes", type=int, default=80)
    args = ap.parse_args()

    path = os.path.join(ROOT, "runs", args.run, "model.pt")
    m, cfg, obs_dim, n_actions = checkpoint.load(path, device="cpu")
    enc, wm = m["encoder"], m["world_model"]

    base = cfg["seed"] + 20_000                           # disjoint from train & eval
    eps = collect(cfg, args.episodes, base)

    # RMS latent norm for normalization
    allz = []
    for O, _ in eps:
        allz.append(enc(torch.as_tensor(O)))
    Z = torch.cat(allz, 0)
    rms = float(Z.pow(2).mean().sqrt())

    recurrent = hasattr(wm, "init_state")

    def step_wm(z1, a, g):
        if recurrent:
            zn, _, g = wm(z1, torch.tensor([a]), g)
            return zn, g
        zn, _ = wm(z1, torch.tensor([a]))
        return zn, None

    one_step, noop = [], []
    drift = {1: [], 3: [], 5: []}
    for O, A in eps:
        z = enc(torch.as_tensor(O))                      # (T+1, D)
        T = len(A)
        for t in range(T):
            zp, _ = step_wm(z[t:t+1], A[t], None)        # fresh state = 1-step
            one_step.append(float((zp[0] - z[t+1]).pow(2).sum().sqrt()))
            noop.append(float((z[t] - z[t+1]).pow(2).sum().sqrt()))
        for k in drift:
            if T >= k:
                zk = z[0:1]
                g = wm.init_state(1, "cpu") if recurrent else None
                for t in range(k):
                    zk, g = step_wm(zk, A[t], g)
                drift[k].append(float((zk[0] - z[k]).pow(2).sum().sqrt()))

    print("\n" + "=" * 56)
    print(f"WM PROBE  run={args.run}  (held-out, {args.episodes} eps)")
    print(f"latent RMS norm                = {rms:.3f}")
    print(f"1-step error                   = {np.mean(one_step):.4f}  "
          f"({100*np.mean(one_step)/rms:.1f}% of RMS)")
    print(f"no-op baseline ||z_t - z_t+1|| = {np.mean(noop):.4f}  "
          f"({100*np.mean(noop)/rms:.1f}% of RMS)")
    ratio = np.mean(one_step) / max(1e-9, np.mean(noop))
    print(f"error / no-op ratio            = {ratio:.3f}  "
          f"(<<1 = real dynamics learned)")
    for k in (1, 3, 5):
        if drift[k]:
            d = np.mean(drift[k])
            print(f"{k}-step rollout drift           = {d:.4f}  ({100*d/rms:.1f}% of RMS)")
    print("=" * 56)


if __name__ == "__main__":
    main()
