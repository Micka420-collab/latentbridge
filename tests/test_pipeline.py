"""End-to-end smoke tests: train -> evaluate -> checkpoint round-trip, for both
the feedforward and the recurrent (rssm) world model."""
import copy
import os

import numpy as np
import torch

from latentbridge import checkpoint
from latentbridge.controller import MPCController
from latentbridge.evaluate import evaluate
from latentbridge.llm import Planner
from latentbridge.train import train


METRIC_KEYS = {"score", "success_rate_guided", "success_rate_unguided",
               "guidance_gain", "align_cosine_heldout"}


def test_train_evaluate_smoke(tiny_cfg):
    m = train(tiny_cfg, verbose=False)
    metrics = evaluate(m, tiny_cfg)
    assert METRIC_KEYS <= set(metrics)
    assert 0.0 <= metrics["success_rate_guided"] <= 1.0
    assert -1.0 <= metrics["align_cosine_heldout"] <= 1.0
    assert np.isfinite(m["history"]).all()


def test_train_evaluate_rssm_smoke(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"]["dynamics"] = "rssm"
    cfg["train"]["seq_len"] = 3
    m = train(cfg, verbose=False)
    metrics = evaluate(m, cfg)
    assert METRIC_KEYS <= set(metrics)


def test_evaluate_deterministic(tiny_cfg):
    m = train(tiny_cfg, verbose=False)
    m1 = evaluate(m, tiny_cfg)
    m2 = evaluate(m, tiny_cfg)
    assert m1["align_cosine_heldout"] == m2["align_cosine_heldout"]
    assert m1["success_rate_guided"] == m2["success_rate_guided"]


def test_checkpoint_round_trip(tiny_cfg, tmp_path):
    m = train(tiny_cfg, verbose=False)
    path = os.path.join(tmp_path, "model.pt")
    checkpoint.save(path, m, tiny_cfg, m["obs_dim"], m["n_actions"])
    m2, cfg2, obs_dim, n_actions = checkpoint.load(path)
    assert cfg2 == tiny_cfg and obs_dim == m["obs_dim"] and n_actions == m["n_actions"]
    x = torch.randn(3, obs_dim)
    with torch.no_grad():
        assert torch.allclose(m["encoder"](x), m2["encoder"](x))


def test_controller_act_and_target_cache(tiny_cfg):
    m = train(tiny_cfg, verbose=False)
    from latentbridge.env import make_env
    env = make_env(tiny_cfg, seed=99)
    obs = env.reset(seed=99)
    planner = Planner(backend="mock", size=tiny_cfg["env"]["size"])
    ctrl = MPCController(m["encoder"], m["world_model"], m["bridge"],
                         m["featurizer"], planner,
                         horizon=3, n_samples=16)
    a = ctrl.act(obs, env=env, guided=False)
    assert 0 <= a < env.action_space
    a = ctrl.act(obs, env=env, guided=True)
    assert 0 <= a < env.action_space
    assert len(ctrl._target_cache) == 1
    ctrl.act(obs, env=env, guided=True)   # same goal -> cache hit, no growth
    assert len(ctrl._target_cache) == 1
