"""Joint training of encoder + world model + decoder + bridge.

Losses (weights live in the config = Arbor's search space):
  L_dyn    next-latent prediction      (self-supervised world model)
  L_rew    reward prediction
  L_recon  observation reconstruction  (grounds the latent)
  L_align  latent->language alignment  (the novel bridge objective)
  L_ground language->latent grounding  (lets the planner pass goals as text)
  L_cycle  bridge invertibility

Env-agnostic: works for any env built by make_env (gridworld, lifeworld, ...).
"""
from __future__ import annotations

import numpy as np
import torch

from .env import make_env
from .build import make_models


def collect(cfg, seed_offset=0):
    """Random-policy rollouts -> transition buffer."""
    obs_l, act_l, rew_l, nobs_l, txt_l = [], [], [], [], []
    rng = np.random.default_rng(cfg["seed"] + seed_offset)
    for ep in range(cfg["train"]["collect_episodes"]):
        env = make_env(cfg, seed=cfg["seed"] + seed_offset + ep)
        obs = env.reset(seed=cfg["seed"] + seed_offset + ep)
        done = False
        while not done:
            txt = env.text_state()
            a = int(rng.integers(0, env.action_space))
            nobs, r, done, _ = env.step(a)
            obs_l.append(obs); act_l.append(a); rew_l.append(r)
            nobs_l.append(nobs); txt_l.append(txt)
            obs = nobs
    return {
        "obs": np.asarray(obs_l, dtype=np.float32),
        "action": np.asarray(act_l, dtype=np.int64),
        "reward": np.asarray(rew_l, dtype=np.float32),
        "next_obs": np.asarray(nobs_l, dtype=np.float32),
        "text": txt_l,
    }


def train(cfg, device="cpu", verbose=True):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    probe = make_env(cfg, seed=cfg["seed"])
    obs_dim = probe.obs_dim
    n_actions = probe.action_space

    m = make_models(cfg, obs_dim, n_actions, device)
    enc, dec, wm, bridge, feat = (m["encoder"], m["decoder"], m["world_model"],
                                  m["bridge"], m["featurizer"])

    params = (list(enc.parameters()) + list(dec.parameters())
              + list(wm.parameters()) + list(bridge.parameters()))
    opt = torch.optim.Adam(params, lr=cfg["train"]["lr"])

    data = collect(cfg)
    n = len(data["action"])
    obs = torch.as_tensor(data["obs"], device=device)
    nobs = torch.as_tensor(data["next_obs"], device=device)
    act = torch.as_tensor(data["action"], device=device)
    rew = torch.as_tensor(data["reward"], device=device)
    lang = torch.as_tensor(feat.embed_batch(data["text"]), device=device)

    w = cfg["loss"]
    bs = cfg["train"]["batch_size"]
    history = []
    rng = np.random.default_rng(cfg["seed"])

    for step in range(cfg["train"]["steps"]):
        idx = torch.as_tensor(rng.integers(0, n, size=min(bs, n)), device=device)
        z = enc(obs[idx])
        with torch.no_grad():
            z_next_tgt = enc(nobs[idx])
        z_next_pred, r_pred = wm(z, act[idx])

        L_dyn = torch.nn.functional.mse_loss(z_next_pred, z_next_tgt)
        L_rew = torch.nn.functional.mse_loss(r_pred, rew[idx])
        L_recon = torch.nn.functional.mse_loss(dec(z), obs[idx])
        L_align = bridge.align_loss(z, lang[idx])
        L_cycle = bridge.cycle_loss(z)
        L_ground = torch.nn.functional.mse_loss(bridge.from_lang(lang[idx]), z.detach())

        loss = (w["dyn"] * L_dyn + w["rew"] * L_rew + w["recon"] * L_recon
                + w["align"] * L_align + w["cycle"] * L_cycle
                + w.get("ground", 0.0) * L_ground)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (step % max(1, cfg["train"]["steps"] // 10) == 0):
            print(f"step {step:5d} | loss {loss.item():.4f} | dyn {L_dyn.item():.4f} "
                  f"rew {L_rew.item():.4f} recon {L_recon.item():.4f} "
                  f"align {L_align.item():.4f} cycle {L_cycle.item():.4f} "
                  f"ground {L_ground.item():.4f}")
        history.append(loss.item())

    m["history"] = history
    m["obs_dim"] = obs_dim
    m["n_actions"] = n_actions
    return m
