"""Reference baselines for the benchmark (BENCHMARKING.md).

To claim "more efficient than what exists", you must compare against the methods
that already exist, in the *same* env and at an *equal budget*. This package holds
those reference baselines:

  - ppo.py        model-free RL (PPO, actor-critic)          -> the classic baseline
  - dqn.py        model-free RL (DQN, value-based)            -> the classic baseline
  - policies.py   random floor + llm_only (planner, no world model)

The world-model-only and the full LatentBridge are produced by the existing
train()/evaluate() with the bridge ablated vs. enabled (see ../benchmark.py).

Every baseline exposes the same contract as the LatentBridge controller:
a policy object with ``act(obs, env) -> int`` evaluated on the SAME held-out seeds.
"""
from .ppo import train_ppo, PPOPolicy
from .dqn import train_dqn, DQNPolicy
from .policies import RandomPolicy, LLMOnlyPolicy, eval_policy

__all__ = [
    "train_ppo", "PPOPolicy",
    "train_dqn", "DQNPolicy",
    "RandomPolicy", "LLMOnlyPolicy", "eval_policy",
]
