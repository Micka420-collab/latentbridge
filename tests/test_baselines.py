"""Fast checks for the comparison harness baselines."""
import copy

from latentbridge.baselines import (LLMOnlyPolicy, RandomPolicy, eval_policy,
                                    train_dqn, train_ppo)


def test_random_policy_eval(tiny_cfg):
    pol = RandomPolicy(4, seed=0)
    metr = eval_policy(pol, tiny_cfg)
    assert {"success_rate", "avg_steps", "avg_reward"} <= set(metr)
    assert 0.0 <= metr["success_rate"] <= 1.0


def test_llm_only_solves_tiny_grid(tiny_cfg):
    # greedy walk to the planner-revealed goal must beat chance on a 3x3 grid
    pol = LLMOnlyPolicy(tiny_cfg, seed=0)
    metr = eval_policy(pol, tiny_cfg)
    assert metr["success_rate"] == 1.0
    assert pol.degraded is False


def test_ppo_dqn_smoke(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    for train_fn in (train_ppo, train_dqn):
        pol, info = train_fn(cfg, seed=0, episodes_budget=3)
        assert info["params"] > 0
        metr = eval_policy(pol, cfg)
        assert 0.0 <= metr["success_rate"] <= 1.0
