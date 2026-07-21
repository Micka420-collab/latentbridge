"""CEM trajectory optimizer: does it actually find good action sequences?

A toy world model with a known optimal action per timestep gives a ground truth
the optimizer must recover — a sharper check than end-to-end success rates.
"""
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from latentbridge.controller import MPCController
from latentbridge.evaluate import evaluate
from latentbridge.train import train

LAT, A, H = 4, 5, 4
TARGET = [3, 1, 4, 0]   # the only rewarded action at each step


class _IdentityEncoder(nn.Module):
    def forward(self, obs):
        return obs[:, :LAT]


class _ToyWM(nn.Module):
    """Latent is untouched; reward 1 iff the action matches TARGET[t]. The step
    index is tracked in the last latent dim so the model stays memoryless."""
    n_actions = A

    def forward(self, z, action):
        t = z[:, -1].long()
        tgt = torch.as_tensor(TARGET, device=z.device)[t.clamp(max=H - 1)]
        reward = (action.long() == tgt).float()
        z_next = z.clone()
        z_next[:, -1] += 1
        return z_next, reward


def _ctrl(optimizer, n_samples=256, seed=0):
    return MPCController(_IdentityEncoder(), _ToyWM(), horizon=H,
                         n_samples=n_samples, gamma=1.0,
                         optimizer=optimizer, seed=seed)


def test_cem_recovers_known_best_sequence():
    obs = np.zeros(LAT, np.float32)
    hits = sum(_ctrl("cem", seed=s).act(obs) == TARGET[0] for s in range(10))
    assert hits == 10


def test_cem_beats_shooting_at_equal_budget_when_sampling_is_hard():
    # tiny budget: uniform shooting rarely samples the 1-in-5^4 best sequence,
    # CEM concentrates probability mass on it across iterations
    obs = np.zeros(LAT, np.float32)
    cem_hits = sum(_ctrl("cem", n_samples=80, seed=s).act(obs) == TARGET[0]
                   for s in range(20))
    shoot_hits = sum(_ctrl("shooting", n_samples=80, seed=s).act(obs) == TARGET[0]
                     for s in range(20))
    assert cem_hits > shoot_hits


def test_controller_deterministic_per_seed():
    obs = np.zeros(LAT, np.float32)
    for opt in ("shooting", "cem"):
        a1 = [_ctrl(opt, seed=7).act(obs) for _ in range(3)]
        a2 = [_ctrl(opt, seed=7).act(obs) for _ in range(3)]
        assert a1[0] == a2[0]
        # a fresh controller with the same seed replays the same draws
        c1, c2 = _ctrl(opt, seed=7), _ctrl(opt, seed=7)
        assert [c1.act(obs) for _ in range(3)] == [c2.act(obs) for _ in range(3)]


def test_unknown_optimizer_rejected():
    with pytest.raises(ValueError):
        _ctrl("gradient-descent")


def test_train_evaluate_cem_smoke(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["control"]["optimizer"] = "cem"
    m = train(cfg, verbose=False)
    metrics = evaluate(m, cfg)
    assert 0.0 <= metrics["success_rate_guided"] <= 1.0
