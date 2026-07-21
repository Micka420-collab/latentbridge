"""Contract tests for every built-in env: shapes, step protocol, language
interface, and per-seed determinism (held-out evaluation depends on it)."""
import numpy as np
import pytest

from latentbridge.env import make_env

KINDS = ["gridworld", "lifeworld", "physicsworld", "desktopworld", "simcalc"]


def _cfg(kind):
    return {"seed": 0, "env": {"kind": kind}}


@pytest.mark.parametrize("kind", KINDS)
def test_env_contract(kind):
    env = make_env(_cfg(kind), seed=3)
    obs = env.reset(seed=3)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (env.obs_dim,)
    assert env.action_space >= 2

    nobs, r, done, info = env.step(0)
    assert nobs.shape == (env.obs_dim,)
    assert isinstance(float(r), float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    assert isinstance(env.text_state(), str) and env.text_state()
    assert isinstance(env.goal_state_text(env.true_goal_key()), str)


@pytest.mark.parametrize("kind", KINDS)
def test_env_deterministic_per_seed(kind):
    rng = np.random.default_rng(0)
    actions = None
    trajs = []
    for _ in range(2):
        env = make_env(_cfg(kind), seed=7)
        obs = env.reset(seed=7)
        if actions is None:
            actions = [int(a) for a in rng.integers(0, env.action_space, size=5)]
        traj = [obs]
        for a in actions:
            obs, _, done, _ = env.step(a)
            traj.append(obs)
            if done:
                break
        trajs.append(np.concatenate(traj))
    assert np.array_equal(trajs[0], trajs[1])


@pytest.mark.parametrize("kind", KINDS)
def test_env_episode_terminates(kind):
    env = make_env(_cfg(kind), seed=1)
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(10_000):
        _, _, done, _ = env.step(int(rng.integers(0, env.action_space)))
        if done:
            return
    pytest.fail("episode never terminated")


def test_gridworld_goal_hidden_vs_visible():
    hidden = make_env({"seed": 0, "env": {"kind": "gridworld", "size": 4}}, seed=0)
    visible = make_env({"seed": 0, "env": {"kind": "gridworld", "size": 4,
                                           "include_goal_in_obs": True}}, seed=0)
    assert hidden.obs_dim == 2 * 16
    assert visible.obs_dim == 3 * 16


def test_gridworld_reach_goal():
    env = make_env({"seed": 0, "env": {"kind": "gridworld", "size": 3,
                                       "max_steps": 30}}, seed=5)
    env.reset(seed=5)
    # walk greedily to the (known) goal; must terminate with reached=True
    info = {}
    for _ in range(30):
        (ar, ac), (gr, gc) = env.agent, env.goal
        if ar != gr:
            a = 1 if gr > ar else 0
        else:
            a = 3 if gc > ac else 2
        _, r, done, info = env.step(a)
        if done:
            break
    assert info.get("reached") is True
    assert r == 1.0


def test_make_env_unknown_kind():
    with pytest.raises(ValueError):
        make_env({"seed": 0, "env": {"kind": "nope"}})
