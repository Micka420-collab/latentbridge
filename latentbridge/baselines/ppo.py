"""PPO — model-free RL baseline (the classic reference).

A small MLP actor-critic trained on-policy with clipped PPO + GAE, consuming the
SAME environment (via make_env) as the world-model methods. The fair budget axis
is *environment episodes*: train_ppo gets exactly ``episodes_budget`` episodes of
interaction, the same number the world-model methods collect. That makes the
comparison a sample-efficiency comparison (BENCHMARKING.md §1).

It sees only the observation — so on a language-only-goal task (goal NOT in obs)
it is structurally blind to where the goal is, which is precisely what the bridge
is meant to fix. Run it in both regimes (goal-in-obs vs language-only) to see that.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..env import make_env


class ACNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class PPOPolicy:
    """Greedy (argmax) policy for held-out evaluation."""

    def __init__(self, net: ACNet, device="cpu"):
        self.net = net.eval()
        self.device = device

    @torch.no_grad()
    def act(self, obs: np.ndarray, env=None) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.net(x)
        return int(torch.argmax(logits, dim=-1).item())


def train_ppo(cfg, seed: int, device="cpu", episodes_budget: int = 80,
              hidden: int = 128, rollout_episodes: int = 8, epochs: int = 4,
              gamma: float = 0.99, lam: float = 0.95, clip: float = 0.2,
              lr: float = 3e-3, ent_coef: float = 0.01, vf_coef: float = 0.5):
    """Train PPO for exactly ``episodes_budget`` env episodes. Returns (policy, info)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe = make_env(cfg, seed=seed)
    obs_dim, n_actions = probe.obs_dim, probe.action_space
    net = ACNet(obs_dim, n_actions, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    ep_used, ep_i = 0, 0
    # training seeds stay disjoint from the held-out eval seeds (cfg.seed + 10_000)
    while ep_used < episodes_budget:
        n_eps = min(rollout_episodes, episodes_budget - ep_used)
        b_obs, b_act, b_logp, b_ret, b_adv = [], [], [], [], []
        for _ in range(n_eps):
            s = seed + 1 + ep_i
            ep_i += 1
            env = make_env(cfg, seed=s)
            obs = env.reset(seed=s)
            obs_l, act_l, logp_l, rew_l, val_l = [], [], [], [], []
            done = False
            while not done:
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, v = net(x)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    logp = dist.log_prob(a)
                obs_l.append(obs)
                act_l.append(int(a.item()))
                logp_l.append(float(logp.item()))
                val_l.append(float(v.item()))
                obs, r, done, _ = env.step(int(a.item()))
                rew_l.append(float(r))
            # GAE-lambda (episode terminates -> bootstrap value 0)
            T = len(rew_l)
            adv = np.zeros(T, dtype=np.float32)
            vals = val_l + [0.0]
            lastgae = 0.0
            for t in reversed(range(T)):
                delta = rew_l[t] + gamma * vals[t + 1] - vals[t]
                lastgae = delta + gamma * lam * lastgae
                adv[t] = lastgae
            ret = adv + np.asarray(val_l, dtype=np.float32)
            b_obs += obs_l
            b_act += act_l
            b_logp += logp_l
            b_ret += list(ret)
            b_adv += list(adv)
        ep_used += n_eps

        bo = torch.as_tensor(np.asarray(b_obs, np.float32), device=device)
        ba = torch.as_tensor(np.asarray(b_act, np.int64), device=device)
        blp = torch.as_tensor(np.asarray(b_logp, np.float32), device=device)
        bret = torch.as_tensor(np.asarray(b_ret, np.float32), device=device)
        badv = torch.as_tensor(np.asarray(b_adv, np.float32), device=device)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)

        for _ in range(epochs):
            logits, v = net(bo)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(ba)
            ratio = torch.exp(logp - blp)
            s1 = ratio * badv
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * badv
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = ((v - bret) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            opt.step()

    info = {"params": sum(p.numel() for p in net.parameters()), "episodes": ep_used}
    return PPOPolicy(net, device), info
