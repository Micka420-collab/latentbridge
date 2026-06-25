"""Held-out evaluation. Produces the metrics Arbor optimizes.

Key comparison: success WITH the LLM/bridge goal guidance vs WITHOUT it. A
positive 'guidance_gain' is direct evidence that the LLM<->world-model coupling
helps. We also report held-out latent<->language alignment.

Env-agnostic via make_env: same code evaluates gridworld and lifeworld.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .env import make_env
from .controller import MPCController
from .llm import Planner


def _run_episodes(ctrl, cfg, guided, seeds):
    succ, steps_used, rewards = [], [], []
    for s in seeds:
        env = make_env(cfg, seed=s)
        obs = env.reset(seed=s)
        done, total, info = False, 0.0, {"reached": False}
        while not done:
            a = ctrl.act(obs, env=env, guided=guided)
            obs, r, done, info = env.step(a)
            total += r
        succ.append(1.0 if info.get("reached") else 0.0)
        steps_used.append(env.t)
        rewards.append(total)
    return float(np.mean(succ)), float(np.mean(steps_used)), float(np.mean(rewards))


def evaluate(modules, cfg, device="cpu"):
    enc, wm, bridge, feat = (modules["encoder"], modules["world_model"],
                             modules["bridge"], modules["featurizer"])
    enc.eval(); wm.eval(); bridge.eval()

    planner = Planner(backend=cfg["planner"]["backend"],
                      model=cfg["planner"].get("model"),
                      size=cfg["env"].get("size", 6))
    ctrl = MPCController(enc, wm, bridge, feat, planner,
                         horizon=cfg["control"]["horizon"],
                         n_samples=cfg["control"]["n_samples"],
                         gamma=cfg["control"]["gamma"],
                         guide_weight=cfg["control"]["guide_weight"],
                         device=device)

    base = cfg["seed"] + 10_000   # held-out seeds disjoint from training
    seeds = list(range(base, base + cfg["eval"]["episodes"]))

    sr_u, st_u, rw_u = _run_episodes(ctrl, cfg, guided=False, seeds=seeds)
    sr_g, st_g, rw_g = _run_episodes(ctrl, cfg, guided=True, seeds=seeds)
    align_cos = _alignment_cosine(enc, bridge, feat, cfg, device, seeds)

    score = sr_g - 0.01 * st_g
    return {
        "score": float(score),
        "success_rate_guided": sr_g,
        "success_rate_unguided": sr_u,
        "guidance_gain": float(sr_g - sr_u),
        "avg_steps_guided": st_g,
        "avg_steps_unguided": st_u,
        "avg_reward_guided": rw_g,
        "align_cosine_heldout": align_cos,
        "final_loss": float(modules["history"][-1]) if modules.get("history") else None,
    }


@torch.no_grad()
def _alignment_cosine(enc, bridge, feat, cfg, device, seeds):
    texts, obss = [], []
    for s in seeds[: min(len(seeds), 64)]:
        env = make_env(cfg, seed=s)
        obs = env.reset(seed=s)
        for _ in range(3):
            texts.append(env.text_state()); obss.append(obs)
            obs, _, done, _ = env.step(np.random.randint(0, env.action_space))
            if done:
                break
    z = enc(torch.as_tensor(np.asarray(obss, dtype=np.float32), device=device))
    proj = F.normalize(bridge.to_lang(z), dim=-1)
    tgt = F.normalize(torch.as_tensor(feat.embed_batch(texts), device=device), dim=-1)
    return float((proj * tgt).sum(-1).mean().item())
