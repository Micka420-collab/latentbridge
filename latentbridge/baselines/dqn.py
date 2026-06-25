"""DQN — model-free RL baseline (value-based reference).

Small MLP Q-network, replay buffer, target network, linear epsilon decay. Same
env, same episode budget as everyone else (sample-efficiency comparison). Like
PPO it only sees the observation, so the language-only-goal regime is where it is
expected to stall near chance — that gap is the bridge's measured contribution.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..env import make_env


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNPolicy:
    """Greedy policy for held-out evaluation."""

    def __init__(self, net: QNet, device="cpu"):
        self.net = net.eval()
        self.device = device

    @torch.no_grad()
    def act(self, obs: np.ndarray, env=None) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(torch.argmax(self.net(x), dim=-1).item())


def train_dqn(cfg, seed: int, device="cpu", episodes_budget: int = 80,
              hidden: int = 128, gamma: float = 0.99, lr: float = 1e-3,
              buffer_size: int = 50_000, batch_size: int = 128,
              target_sync: int = 200, warmup: int = 256,
              eps_start: float = 1.0, eps_end: float = 0.05):
    """Train DQN for exactly ``episodes_budget`` env episodes. Returns (policy, info)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    probe = make_env(cfg, seed=seed)
    obs_dim, n_actions = probe.obs_dim, probe.action_space

    q = QNet(obs_dim, n_actions, hidden).to(device)
    tgt = QNet(obs_dim, n_actions, hidden).to(device)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    # O(1)-indexed ring buffer (deque indexing is O(n) — too slow to sample from)
    buf, cap = [], buffer_size
    write = 0
    total_steps_est = max(1, episodes_budget * int(cfg["env"].get("max_steps", 50)))
    gstep = 0

    for ep_i in range(episodes_budget):
        s = seed + 1 + ep_i
        env = make_env(cfg, seed=s)
        obs = env.reset(seed=s)
        done = False
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * gstep / total_steps_est)
            if rng.random() < eps:
                a = int(rng.integers(0, n_actions))
            else:
                with torch.no_grad():
                    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    a = int(torch.argmax(q(x), dim=-1).item())
            nobs, r, done, _ = env.step(a)
            trans = (obs, a, float(r), nobs, float(done))
            if len(buf) < cap:
                buf.append(trans)
            else:
                buf[write] = trans
                write = (write + 1) % cap
            obs = nobs
            gstep += 1

            if len(buf) >= max(warmup, batch_size):
                idx = rng.integers(0, len(buf), size=batch_size)
                batch = [buf[i] for i in idx]
                bo = torch.as_tensor(np.asarray([b[0] for b in batch], np.float32), device=device)
                ba = torch.as_tensor(np.asarray([b[1] for b in batch], np.int64), device=device)
                br = torch.as_tensor(np.asarray([b[2] for b in batch], np.float32), device=device)
                bn = torch.as_tensor(np.asarray([b[3] for b in batch], np.float32), device=device)
                bd = torch.as_tensor(np.asarray([b[4] for b in batch], np.float32), device=device)
                qv = q(bo).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nq = tgt(bn).max(1).values
                    target = br + gamma * (1.0 - bd) * nq
                loss = ((qv - target) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if gstep % target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

    info = {"params": sum(p.numel() for p in q.parameters()), "episodes": episodes_budget}
    return DQNPolicy(q, device), info
