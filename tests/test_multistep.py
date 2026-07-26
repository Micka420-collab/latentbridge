"""train.multistep_k: free-running K-step consistency training for the
feedforward world model (the regime MPC actually uses at plan time)."""
import copy

import numpy as np

from latentbridge.evaluate import evaluate
from latentbridge.train import train


def test_multistep_feedforward_smoke(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["train"]["multistep_k"] = 3
    m = train(cfg, verbose=False)
    assert not hasattr(m["world_model"], "init_state")   # still the feedforward model
    assert np.isfinite(m["history"]).all()
    metrics = evaluate(m, cfg)
    assert 0.0 <= metrics["success_rate_guided"] <= 1.0


def test_multistep_k1_uses_historical_onestep_path(tiny_cfg):
    # k=1 (and absent) must train through the exact historical 1-step branch:
    # same losses drawn from the same RNG stream -> identical loss history
    m_absent = train(copy.deepcopy(tiny_cfg), verbose=False)
    cfg1 = copy.deepcopy(tiny_cfg)
    cfg1["train"]["multistep_k"] = 1
    m_k1 = train(cfg1, verbose=False)
    assert m_absent["history"] == m_k1["history"]


def test_multistep_still_works_with_rssm(tiny_cfg):
    # rssm keeps its own seq_len regime regardless of multistep_k
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"]["dynamics"] = "rssm"
    cfg["train"]["seq_len"] = 3
    cfg["train"]["multistep_k"] = 4
    m = train(cfg, verbose=False)
    assert hasattr(m["world_model"], "init_state")
    assert np.isfinite(m["history"]).all()
