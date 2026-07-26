"""Featurizer factory + the real-embeddings backend (optional dependency)."""
import copy

import numpy as np
import pytest

from latentbridge.build import make_models
from latentbridge.text_features import HashingTextFeaturizer, make_featurizer

st = pytest.importorskip if False else None  # noqa: F841 (marker helper below)

try:
    import sentence_transformers  # noqa: F401
    HAS_ST = True
except ImportError:
    HAS_ST = False


def _cfg(**model_extra):
    return {"seed": 0,
            "model": {"latent_dim": 8, "lang_dim": 16, "hidden": 16, **model_extra}}


def test_factory_default_is_hashing_with_config_knobs():
    f = make_featurizer(_cfg(text_ngram=2))
    assert isinstance(f, HashingTextFeaturizer)
    assert f.dim == 16 and f.ngram == 2 and f.seed == 0


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_featurizer(_cfg(text_features="word2vec"))


@pytest.mark.skipif(not HAS_ST, reason="sentence-transformers not installed")
def test_stransformer_backend_mechanics():
    # NOTE: deliberately pins only the MECHANICS (norm, cache, batch), not the
    # semantic geometry — measured (2026-07): small ST models embed transposed
    # coordinates ("row 2 col 3" vs "row 3 col 2") at cosine ~1.0, so semantic
    # assertions about goal discrimination would pin a property the model does
    # NOT have. See scripts/paraphrase_probe.py and the PR for the numbers.
    f = make_featurizer(_cfg(text_features="stransformer"))
    assert f.dim > 0
    canon = f.embed("Grid 6x6. Agent at row 2 col 3.")
    assert abs(float(np.linalg.norm(canon)) - 1.0) < 1e-4
    assert np.array_equal(canon, f.embed("Grid 6x6. Agent at row 2 col 3."))  # cached
    # batch path fills the same cache
    batch = f.embed_batch(["Grid 6x6. Agent at row 2 col 3.", "hello world"])
    assert np.array_equal(batch[0], canon)


@pytest.mark.skipif(not HAS_ST, reason="sentence-transformers not installed")
def test_bridge_lang_dim_follows_featurizer(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"]["text_features"] = "stransformer"
    m = make_models(cfg, obs_dim=18, n_actions=4)
    feat = m["featurizer"]
    assert feat.dim != cfg["model"]["lang_dim"]          # 384 vs 24: must not clash
    out = m["bridge"].to_lang(__import__("torch").randn(2, cfg["model"]["latent_dim"]))
    assert out.shape == (2, feat.dim)
