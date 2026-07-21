import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def tiny_cfg():
    """Smallest config that exercises the full train -> evaluate pipeline."""
    return {
        "seed": 0,
        "env": {
            "kind": "gridworld",
            "size": 3,
            "max_steps": 12,
            "n_walls": 0,
            "step_penalty": 0.01,
            "include_goal_in_obs": False,
        },
        "model": {"latent_dim": 16, "lang_dim": 24, "hidden": 32},
        "loss": {"dyn": 1.0, "rew": 1.0, "recon": 0.5, "align": 0.5,
                 "cycle": 0.1, "ground": 0.5},
        "train": {"collect_episodes": 4, "steps": 30, "batch_size": 32, "lr": 1e-3},
        "control": {"horizon": 3, "n_samples": 32, "gamma": 0.95, "guide_weight": 1.0},
        "planner": {"backend": "mock"},
        "eval": {"episodes": 3},
    }
