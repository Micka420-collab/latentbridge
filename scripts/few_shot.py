"""Few-shot / 'learn how to learn an app' — does pretraining on several apps let
the world model learn a NEVER-SEEN app from only a handful of examples?

We use 3 calculator 'dialects' (same structure, different press->display dynamics):
  A: digit k -> k        B: digit k -> (k+3)%10        C: digit k -> 9-k  (held-out)
Train a base world model on A+B, then ADAPT to C with only K examples, starting
either from the pretrained weights or from scratch — and compare. Lower error
from the pretrained start = it learned HOW to learn a new app.

(Clean, controlled demonstration of the mechanism. The same harness applies to
REAL apps as the all-day heartbeat accumulates them.)

    python scripts/few_shot.py --k 24
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latentbridge.models import Encoder, Decoder, WorldModel  # noqa: E402

N_ACT = 11


def onehot(v):
    x = np.zeros(10, dtype=np.float32); x[int(v) % 10] = 1.0; return x


def gen(dialect, n, seed):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        v = int(rng.integers(0, 10))
        a = int(rng.integers(0, N_ACT))
        nv = 0 if a == 0 else dialect[a - 1]
        data.append((onehot(v), a, onehot(nv)))
    return data


def build():
    enc = Encoder(10, 16, 64); dec = Decoder(16, 10, 64); wm = WorldModel(16, N_ACT, 64)
    return enc, dec, wm


def train(models, data, epochs, lr=1e-3, seed=0):
    enc, dec, wm = models
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(wm.parameters()), lr=lr)
    rng = np.random.default_rng(seed)
    X0 = torch.as_tensor(np.stack([d[0] for d in data]))
    A = torch.as_tensor(np.array([d[1] for d in data]))
    X1 = torch.as_tensor(np.stack([d[2] for d in data]))
    n = len(data)
    for _ in range(epochs):
        idx = torch.as_tensor(rng.integers(0, n, size=min(64, n)))
        z = enc(X0[idx])
        with torch.no_grad():
            z1 = enc(X1[idx])
        zp, _ = wm(z, A[idx])
        loss = torch.nn.functional.mse_loss(zp, z1) + torch.nn.functional.mse_loss(dec(z), X0[idx])
        opt.zero_grad(); loss.backward(); opt.step()


def evaluate(models, data):
    enc, dec, wm = models
    enc.eval(); dec.eval(); wm.eval()
    with torch.no_grad():
        err, noop = [], []
        for f0, a, f1 in data:
            z = enc(torch.as_tensor(f0).unsqueeze(0))
            pred = dec(wm(z, torch.tensor([a]))[0])[0].numpy()
            err.append(float(np.linalg.norm(pred - f1)))
            noop.append(float(np.linalg.norm(f0 - f1)))
    enc.train(); dec.train(); wm.train()
    return float(np.mean(err)), float(np.mean(noop)), float(np.mean(err) / max(1e-9, np.mean(noop)))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=24, help="few-shot examples of the new app")
    args = ap.parse_args()
    torch.manual_seed(0)

    A = list(range(10))
    B = [(k + 3) % 10 for k in range(10)]
    C = [9 - k for k in range(10)]                 # held-out app

    base_data = gen(A, 400, 1) + gen(B, 400, 2)
    c_adapt = gen(C, args.k, 3)
    c_eval = gen(C, 300, 4)

    print(f"=== FEW-SHOT : apprendre une app jamais vue (C: k->9-k), K={args.k} ===\n")

    base = build(); train(base, base_data, 2500)
    print(f"  base (entraînée sur A+B) — ZERO-SHOT sur C : ratio={evaluate(base, c_eval)[2]:.2f}  (ne connaît pas C)")

    pre = copy.deepcopy(base); train(pre, c_adapt, 400)
    e_pre, _, r_pre = evaluate(pre, c_eval)

    scr = build(); train(scr, c_adapt, 400)
    e_scr, _, r_scr = evaluate(scr, c_eval)

    print(f"  PRÉ-ENTRAÎNÉE + {args.k} exemples de C : err={e_pre:.3f}  ratio={r_pre:.2f}")
    print(f"  FROM-SCRATCH  + {args.k} exemples de C : err={e_scr:.3f}  ratio={r_scr:.2f}")
    gain = (e_scr - e_pre) / max(1e-9, e_scr)
    verdict = ("✅ le pré-entraînement aide à apprendre la nouvelle app"
               if e_pre < e_scr * 0.9 else
               ("≈ effet marginal" if e_pre < e_scr else "❌ pas d'aide"))
    print(f"\n  => erreur {100*gain:+.0f}% vs from-scratch   {verdict}")


if __name__ == "__main__":
    main()
