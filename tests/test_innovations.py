"""Bigram featurizer (model.text_ngram) + anti-cycling revisit penalty
(control.revisit_weight): mechanism-level tests for both innovations."""
import copy

import numpy as np
import torch

from latentbridge.controller import MPCController
from latentbridge.evaluate import evaluate
from latentbridge.text_features import HashingTextFeaturizer
from latentbridge.train import train


# --------------------------- text_ngram --------------------------------- #
def test_unigram_aliases_transposed_cells_bigram_does_not():
    a = "Grid 6x6. Agent at row 2 col 3."
    b = "Grid 6x6. Agent at row 3 col 2."
    uni = HashingTextFeaturizer(dim=64, seed=0, ngram=1)
    # the measured defect: same token multiset -> same embedding (up to fp
    # summation order), so the bridge cannot tell (2,3) from (3,2)
    assert float(uni.embed(a) @ uni.embed(b)) > 1.0 - 1e-6
    bi = HashingTextFeaturizer(dim=64, seed=0, ngram=2)
    assert float(bi.embed(a) @ bi.embed(b)) < 0.95


def test_ngram1_is_bit_identical_to_default():
    texts = ["Grid 6x6. Agent at row 1 col 4.", "reach the goal", ""]
    default = HashingTextFeaturizer(dim=32, seed=3)
    explicit = HashingTextFeaturizer(dim=32, seed=3, ngram=1)
    for t in texts:
        assert np.array_equal(default.embed(t), explicit.embed(t))


def test_bigram_embedding_contract():
    f = HashingTextFeaturizer(dim=48, seed=1, ngram=2)
    v = f.embed("agent at row 1 col 2")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    assert np.array_equal(v, f.embed("agent at row 1 col 2"))
    assert np.array_equal(f.embed("solo"),
                          HashingTextFeaturizer(dim=48, seed=1, ngram=1).embed("solo"))
    assert np.array_equal(f.embed(""), np.zeros(48, np.float32))


# ------------------------- revisit penalty ------------------------------ #
LAT = 6


class _IdEnc(torch.nn.Module):
    def forward(self, obs):
        return obs[:, :LAT]


class _StaticWM(torch.nn.Module):
    """Latent unchanged by any action, zero reward — isolates the shaping terms."""
    n_actions = 3

    def forward(self, z, action):
        return z + torch.nn.functional.one_hot(action.long(), LAT).float(), torch.zeros(z.shape[0])


def _controller(revisit_weight):
    return MPCController(_IdEnc(), _StaticWM(), horizon=2, n_samples=16,
                         revisit_weight=revisit_weight, revisit_mem=4, seed=0)


def test_score_penalizes_sequences_near_visited_states():
    ctrl = _controller(1.0)
    target = torch.zeros(1, LAT) + 5.0
    visited = torch.zeros(LAT); visited[0] = 1.0   # "we just were at" +e0
    ctrl._visited.append(visited)
    z0 = torch.zeros(1, LAT)
    toward_visited = torch.zeros(2, 2, dtype=torch.long)          # action 0 twice -> +e0 twice
    away = torch.ones(2, 2, dtype=torch.long)                     # action 1 twice -> +e1 twice
    scores = ctrl._score(z0, torch.cat([toward_visited[:1], away[:1]]), target)
    assert scores[1] > scores[0]  # revisiting scores strictly worse

    # weight 0 -> the two sequences tie (both equidistant from the target)
    ctrl0 = _controller(0.0)
    scores0 = ctrl0._score(z0, torch.cat([toward_visited[:1], away[:1]]), target)
    assert torch.allclose(scores0[0], scores0[1])


def test_memory_guided_only_and_cleared_on_new_episode():
    ctrl = _controller(1.0)

    class _Env:
        t = 0
    obs = np.zeros(LAT, np.float32)
    ctrl.act(obs, env=_Env(), guided=False)
    assert len(ctrl._visited) == 0            # unguided never records
    ctrl._visited.append(torch.zeros(LAT))
    ctrl.act(obs, env=_Env(), guided=False)   # env.t == 0 -> memory cleared
    assert len(ctrl._visited) == 0


def test_revisit_weight_zero_is_bit_identical():
    # same seed, same obs: default controller and revisit_weight=0 pick the
    # same action with identical RNG consumption
    torch.manual_seed(0)
    obs = np.random.default_rng(0).standard_normal(LAT).astype(np.float32)
    a_default = MPCController(_IdEnc(), _StaticWM(), horizon=3, n_samples=32, seed=5).act(obs)
    a_zero = _controller(0.0)
    a_zero.horizon, a_zero.n_samples = 3, 32
    a_zero._gen.manual_seed(5)
    assert a_default == a_zero.act(obs)


# --------------------------- end-to-end --------------------------------- #
def test_train_evaluate_with_both_innovations(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"]["text_ngram"] = 2
    cfg["control"]["revisit_weight"] = 1.0
    m = train(cfg, verbose=False)
    assert m["featurizer"].ngram == 2
    metrics = evaluate(m, cfg)
    assert 0.0 <= metrics["success_rate_guided"] <= 1.0
