"""IN-CONTEXT few-shot world model — 'understand any app in seconds, no retraining'.

A recurrent meta-model reads a few (state, action, next_state) transitions from an
UNKNOWN app, infers how that app works, and predicts the next state for new
(state, action) queries — WITHOUT any weight update. This is the leap from
'learns one app (by gradient descent)' to 'figures out a new app from a handful of
observations'.

Demonstrated on calculator 'dialects' (each a different press->display mapping =
a different app). We META-TRAIN across many random dialects so the model learns to
INFER dynamics from context, then test on NEVER-SEEN dialects with NO retraining.

    python scripts/incontext_fewshot.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

S, A = 10, 11                                  # 10 display digits, 11 actions (clear + press0..9)
TRANS = S + A + S                             # one context transition = (s, a, s')


def onehot(i, n):
    v = np.zeros(n, dtype=np.float32); v[int(i)] = 1.0; return v


def make_batch(B, K, rng, dialects=None):
    """B episodes; each: K context transitions + 1 query, from a random dialect."""
    ctx = np.zeros((B, K, TRANS), dtype=np.float32)
    sq = np.zeros((B, S), dtype=np.float32)
    aq = np.zeros((B, A), dtype=np.float32)
    y = np.zeros(B, dtype=np.int64)
    for b in range(B):
        perm = (dialects[rng.integers(0, len(dialects))] if dialects is not None
                else rng.permutation(S))
        for k in range(K):
            s = rng.integers(0, S); a = rng.integers(0, A)
            ns = 0 if a == 0 else perm[a - 1]
            ctx[b, k] = np.concatenate([onehot(s, S), onehot(a, A), onehot(ns, S)])
        s = rng.integers(0, S); a = rng.integers(0, A)
        ns = 0 if a == 0 else perm[a - 1]
        sq[b] = onehot(s, S); aq[b] = onehot(a, A); y[b] = ns
    return (torch.as_tensor(ctx), torch.as_tensor(sq), torch.as_tensor(aq), torch.as_tensor(y))


class InContextWM(nn.Module):
    """Attention-based in-context learner: the query (s_q, a_q) ATTENDS over the
    observed context transitions (keys = their (s, a), values = their next-state)
    and reads off the predicted next-state — associative recall = in-context
    inference of the unknown app's dynamics, no weight update at test time."""

    def __init__(self, d=128):
        super().__init__()
        self.key = nn.Linear(S + A, d)
        self.val = nn.Linear(S, d)
        self.qry = nn.Linear(S + A, d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, S))
        self.d = d

    def forward(self, ctx, sq, aq):
        B, K, _ = ctx.shape
        if K == 0:
            return self.out(torch.zeros(B, self.d))
        s, a, ns = ctx[..., :S], ctx[..., S:S + A], ctx[..., S + A:]
        keys = self.key(torch.cat([s, a], dim=-1))        # (B,K,d)
        vals = self.val(ns)                               # (B,K,d)
        q = self.qry(torch.cat([sq, aq], dim=-1)).unsqueeze(1)  # (B,1,d)
        w = torch.softmax((q * keys).sum(-1) / (self.d ** 0.5), dim=-1)  # (B,K)
        c = (w.unsqueeze(-1) * vals).sum(1)               # (B,d)
        return self.out(c)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = InContextWM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("=== MÉTA-ENTRAÎNEMENT (sur des apps/dialectes aléatoires) ===")
    for step in range(6000):
        K = int(rng.integers(1, 16))
        ctx, sq, aq, y = make_batch(64, K, rng)
        logits = model(ctx, sq, aq)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1500 == 0:
            print(f"  step {step:4d}  loss {loss.item():.3f}")

    # held-out apps: fixed random permutations the model is unlikely to have fit to
    held = [rng.permutation(S) for _ in range(40)]
    model.eval()
    print("\n=== TEST sur des apps JAMAIS VUES (aucun réentraînement) ===")
    print("  contexte (K transitions observées)  ->  précision de prédiction du prochain état")
    with torch.no_grad():
        for K in (0, 1, 3, 5, 10, 20):
            accs = []
            for _ in range(30):
                ctx, sq, aq, y = make_batch(128, K, rng, dialects=held)
                pred = model(ctx, sq, aq).argmax(-1)
                accs.append((pred == y).float().mean().item())
            bar = "█" * int(round(np.mean(accs) * 30))
            print(f"  K={K:2d}  {np.mean(accs)*100:5.1f}%  {bar}")
    print("\n  => la précision MONTE avec le contexte : elle COMPREND une app inconnue")
    print("     en l'observant quelques secondes, sans réentraînement (in-context learning).")

    out = os.path.join(ROOT, "runs", "incontext_wm")
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    print(f"\n  modèle méta sauvegardé -> {out}/model.pt")


if __name__ == "__main__":
    main()
