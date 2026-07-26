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


def collect_sequences(cfg, L, seed_offset=0):
    """Random-policy rollouts sliced into fixed windows of L transitions
    (L+1 observations) — for multi-step training of the recurrent world model."""
    rng = np.random.default_rng(cfg["seed"] + 777 + seed_offset)
    O, A, R = [], [], []
    for ep in range(cfg["train"]["collect_episodes"]):
        env = make_env(cfg, seed=cfg["seed"] + seed_offset + ep)
        obs = env.reset(seed=cfg["seed"] + seed_offset + ep)
        ob_seq, ac_seq, rw_seq = [obs], [], []
        done = False
        while not done:
            a = int(rng.integers(0, env.action_space))
            nobs, r, done, _ = env.step(a)
            ac_seq.append(a); rw_seq.append(r); ob_seq.append(nobs)
            obs = nobs
        for s in range(0, len(ac_seq) - L + 1):
            O.append(ob_seq[s:s + L + 1])
            A.append(ac_seq[s:s + L])
            R.append(rw_seq[s:s + L])
    if not O:
        raise ValueError(f"no length-{L} windows collected; lower train.seq_len")
    return (np.asarray(O, np.float32), np.asarray(A, np.int64),
            np.asarray(R, np.float32))


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
    recurrent = cfg["model"].get("dynamics") == "rssm"
    # train.multistep_k >= 2 turns on free-running K-step consistency training
    # for the FEEDFORWARD model too (the same regime the rssm branch always
    # uses): the model must predict K steps from its own outputs, which is
    # exactly what MPC asks of it at plan time. Default 1 = historical 1-step.
    multistep_k = int(cfg["train"].get("multistep_k", 1))
    seq_train = recurrent or multistep_k > 1

    if seq_train:
        # multi-step (free-running) dynamics training; per-obs losses
        # (recon/align/cycle/ground) stay identical.
        L = int(cfg["train"].get("seq_len", 6)) if recurrent else multistep_k
        Os, As, Rs = collect_sequences(cfg, L)
        Os = torch.as_tensor(Os, device=device)            # (N, L+1, obs_dim)
        As = torch.as_tensor(As, device=device)            # (N, L)
        Rs = torch.as_tensor(Rs, device=device)            # (N, L)
        nseq = As.shape[0]

    for step in range(cfg["train"]["steps"]):
        idx = torch.as_tensor(rng.integers(0, n, size=min(bs, n)), device=device)
        z = enc(obs[idx])
        L_recon = torch.nn.functional.mse_loss(dec(z), obs[idx])
        L_align = bridge.align_loss(z, lang[idx])
        L_cycle = bridge.cycle_loss(z)
        L_ground = torch.nn.functional.mse_loss(bridge.from_lang(lang[idx]), z.detach())

        if seq_train:
            sidx = torch.as_tensor(rng.integers(0, nseq, size=min(bs, nseq)), device=device)
            ob, ac, rw = Os[sidx], As[sidx], Rs[sidx]      # (B,L+1,od),(B,L),(B,L)
            B = ob.shape[0]
            with torch.no_grad():
                z_tgt = enc(ob.reshape(-1, obs_dim)).reshape(B, L + 1, -1)
            z_pred = enc(ob[:, 0])
            g = wm.init_state(B, device) if recurrent else None
            L_dyn = z_pred.new_zeros(())
            L_rew = z_pred.new_zeros(())
            for t in range(L):
                if recurrent:
                    z_pred, r_pred, g = wm(z_pred, ac[:, t], g)
                else:
                    z_pred, r_pred = wm(z_pred, ac[:, t])
                L_dyn = L_dyn + torch.nn.functional.mse_loss(z_pred, z_tgt[:, t + 1])
                L_rew = L_rew + torch.nn.functional.mse_loss(r_pred, rw[:, t])
            L_dyn = L_dyn / L
            L_rew = L_rew / L
        else:
            with torch.no_grad():
                z_next_tgt = enc(nobs[idx])
            z_next_pred, r_pred = wm(z, act[idx])
            L_dyn = torch.nn.functional.mse_loss(z_next_pred, z_next_tgt)
            L_rew = torch.nn.functional.mse_loss(r_pred, rew[idx])

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
