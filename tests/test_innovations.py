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


def test_bigrams_are_directional():
    # structural order pin: "a b" and "b a" share the token multiset and would
    # be identical under order-INSENSITIVE (sorted-pair) bigrams; directional
    # bigrams hash a_b vs b_a to unrelated vectors
    f = HashingTextFeaturizer(dim=64, seed=0, ngram=2)
    assert float(f.embed("row col") @ f.embed("col row")) < 0.999


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


def test_score_weight_zero_matches_historical_formula():
    # pins the historical semantics structurally: with revisit_weight=0 and a
    # NON-EMPTY memory, _score must equal the reference formula
    # sum_t gamma^t * (r_t - guide_weight * ||z_t - z*||^2_mean) computed here
    # independently for the toy dynamics (StaticWM: z accumulates one-hots, r=0)
    ctrl = _controller(0.0)
    ctrl._visited.append(torch.zeros(LAT))
    target = torch.full((1, LAT), 5.0)
    actions = torch.tensor([[0, 1], [2, 2]])
    scores = ctrl._score(torch.zeros(1, LAT), actions, target)
    for k in range(2):
        z = torch.zeros(LAT)
        expected = 0.0
        for t in range(2):
            z = z + torch.nn.functional.one_hot(actions[k, t], LAT).float()
            expected -= ctrl.guide_weight * (ctrl.gamma ** t) * float(((z - target[0]) ** 2).mean())
        assert abs(float(scores[k]) - expected) < 1e-5


def test_nearest_visited_state_dominates_and_deque_evicts():
    ctrl = _controller(1.0)
    far = torch.full((LAT,), 9.0)
    near = torch.zeros(LAT); near[1] = 2.0        # close to where "away" goes
    ctrl._visited.append(far)
    ctrl._visited.append(near)
    target = torch.full((1, LAT), 5.0)
    toward_near = torch.ones(1, 2, dtype=torch.long)    # +e1 twice -> approaches `near`
    toward_free = torch.full((1, 2), 2, dtype=torch.long)
    scores = ctrl._score(torch.zeros(1, LAT),
                         torch.cat([toward_near, toward_free]), target)
    # max(1) semantics: proximity to ANY memory entry (here `near`) must bite
    # even though `far` is remote
    assert scores[1] > scores[0]

    for i in range(10):                            # maxlen=4 in _controller
        ctrl._visited.append(torch.full((LAT,), float(i)))
    assert len(ctrl._visited) == 4
    assert float(ctrl._visited[-1][0]) == 9.0


class _EnvStub:
    def __init__(self, t):
        self.t = t


class _PlannerStub:
    def propose(self, env):
        return {"subgoal_text": "the goal"}


class _FeaturizerStub:
    def embed(self, text):
        return np.ones(LAT, np.float32)


class _BridgeStub(torch.nn.Module):
    def from_lang(self, e):
        return e * 5.0


def _guided_controller(revisit_weight, optimizer="shooting"):
    return MPCController(_IdEnc(), _StaticWM(), bridge=_BridgeStub(),
                         featurizer=_FeaturizerStub(), planner=_PlannerStub(),
                         horizon=2, n_samples=16, optimizer=optimizer,
                         cem_iters=2, revisit_weight=revisit_weight,
                         revisit_mem=4, seed=0)


def test_guided_act_records_memory_and_persists_within_episode():
    ctrl = _guided_controller(1.0)
    obs = np.zeros(LAT, np.float32)
    ctrl.act(obs, env=_EnvStub(t=0), guided=True)
    ctrl.act(obs, env=_EnvStub(t=1), guided=True)
    ctrl.act(obs, env=_EnvStub(t=2), guided=True)
    assert len(ctrl._visited) == 3            # real act() path fills the memory
    ctrl.act(obs, env=_EnvStub(t=0), guided=True)
    assert len(ctrl._visited) == 1            # new episode -> cleared, then z0


def test_memory_guided_only_and_cleared_on_new_episode():
    ctrl = _controller(1.0)
    obs = np.zeros(LAT, np.float32)
    ctrl.act(obs, env=_EnvStub(t=0), guided=False)
    assert len(ctrl._visited) == 0            # unguided never records
    ctrl._visited.append(torch.zeros(LAT))
    ctrl.act(obs, env=_EnvStub(t=0), guided=False)   # env.t == 0 -> memory cleared
    assert len(ctrl._visited) == 0


def test_cem_with_revisit_penalty_returns_valid_action():
    ctrl = _guided_controller(1.0, optimizer="cem")
    obs = np.zeros(LAT, np.float32)
    for t in range(4):
        a = ctrl.act(obs, env=_EnvStub(t=t), guided=True)
        assert 0 <= a < _StaticWM.n_actions
    # deterministic per seed even with the batch-normalized penalty
    ctrl2 = _guided_controller(1.0, optimizer="cem")
    for t in range(4):
        assert ctrl2.act(obs, env=_EnvStub(t=t), guided=True) is not None
    ctrl3, ctrl4 = _guided_controller(1.0, "cem"), _guided_controller(1.0, "cem")
    seq3 = [ctrl3.act(obs, env=_EnvStub(t=t), guided=True) for t in range(4)]
    seq4 = [ctrl4.act(obs, env=_EnvStub(t=t), guided=True) for t in range(4)]
    assert seq3 == seq4


def test_revisit_weight_zero_is_inert_with_populated_memory():
    # like-for-like: two controllers with the SAME seed, one carrying a stuffed
    # memory at weight 0 — the memory must not influence the sampled actions
    obs = np.random.default_rng(0).standard_normal(LAT).astype(np.float32)
    a = MPCController(_IdEnc(), _StaticWM(), horizon=3, n_samples=32, seed=5)
    b = MPCController(_IdEnc(), _StaticWM(), horizon=3, n_samples=32, seed=5,
                      revisit_weight=0.0, revisit_mem=4)
    b._visited.append(torch.full((LAT,), 3.0))
    tgt = torch.full((1, LAT), 5.0)
    z0 = torch.zeros(1, LAT)
    acts = torch.randint(0, 3, (8, 3), generator=torch.Generator().manual_seed(1))
    assert torch.allclose(a._score(z0, acts, tgt), b._score(z0, acts, tgt))
    assert a.act(obs) == b.act(obs)


# --------------------------- end-to-end --------------------------------- #
def test_train_evaluate_with_both_innovations(tiny_cfg, monkeypatch):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"]["text_ngram"] = 2
    cfg["control"]["revisit_weight"] = 1.0
    cfg["control"]["revisit_mem"] = 4
    m = train(cfg, verbose=False)
    assert m["featurizer"].ngram == 2

    # pin the evaluate.py pass-through: the controller must actually receive
    # the knobs the search loop tunes
    import latentbridge.evaluate as ev
    captured = {}

    class _Spy(MPCController):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ev, "MPCController", _Spy)
    metrics = ev.evaluate(m, cfg)
    assert captured["revisit_weight"] == 1.0
    assert captured["revisit_mem"] == 4
    assert 0.0 <= metrics["success_rate_guided"] <= 1.0
