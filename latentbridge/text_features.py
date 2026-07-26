"""Language-space provider.

For fast, fully-offline training we need a *language embedding target* for every
state at every step. Calling a remote embedding API in the hot training loop is
too slow/expensive, so the default is a deterministic local featurizer (hashed
bag-of-tokens). It is a faithful stand-in: a fixed semantic space the bridge
learns to align to.

SWAP POINT: to use real LLM embeddings, implement `embed()` with cached
OpenRouter / local sentence-transformer vectors and set lang_dim accordingly.
The architecture does not change — only the target vectors do.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingTextFeaturizer:
    """ngram=1 is the historical bag-of-tokens. ngram=2 additionally hashes
    adjacent-token bigrams, making the space ORDER-AWARE: with unigrams alone,
    "row 2 col 3" and "row 3 col 2" embed bit-identically (same token multiset),
    so the bridge is trained toward two different states at once. Bigrams
    ("row_2", "2_col", "col_3") break that aliasing while staying deterministic
    and offline — and are a more faithful stand-in for real (order-aware) LLM
    embeddings at this swap point."""

    def __init__(self, dim: int = 128, seed: int = 0, ngram: int = 1):
        self.dim = int(dim)
        self.seed = int(seed)
        self.ngram = int(ngram)
        # token -> vector memo: the sha256 + RNG expansion is deterministic per
        # token, and embed_batch calls it for every token occurrence over
        # thousands of near-identical state descriptions.
        self._cache: dict[str, np.ndarray] = {}

    def _token_vec(self, tok: str) -> np.ndarray:
        v = self._cache.get(tok)
        if v is None:
            h = hashlib.sha256(f"{self.seed}:{tok}".encode()).digest()
            # expand the digest deterministically to dim floats in [-1, 1]
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            v = rng.standard_normal(self.dim).astype(np.float32)
            self._cache[tok] = v
        return v

    def embed(self, text: str) -> np.ndarray:
        toks = _TOKEN.findall(text.lower())
        if not toks:
            return np.zeros(self.dim, dtype=np.float32)
        feats = list(toks)
        if self.ngram >= 2:
            feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.mean([self._token_vec(t) for t in feats], axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_batch(self, texts) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts], axis=0)


class SentenceTransformerFeaturizer:
    """REAL language embeddings at the swap point (model.text_features:
    stransformer). Unlike the hashing stand-in, these are paraphrase-robust:
    "Reach row 2 col 3" and "the agent should end up at row 2, column 3" land
    close together, so a frozen LLM can phrase goals freely instead of having
    to emit the env's canonical text.

    The embedding model is FROZEN (never trained) — it plays the same role as
    the frozen LLM: a fixed semantic space the bridge learns to align to.
    Embeddings are cached by exact text; the built-in envs have a small set of
    distinct state texts, so training cost is one batch-encode per unique text.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",
                 device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # optional dep
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension",
                          self.model.get_sentence_embedding_dimension)
        self.dim = int(get_dim())
        self._cache: dict[str, np.ndarray] = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = self.model.encode(texts, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        return out.astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        v = self._cache.get(text)
        if v is None:
            v = self._encode([text])[0]
            self._cache[text] = v
        return v

    def embed_batch(self, texts) -> np.ndarray:
        missing = sorted({t for t in texts if t not in self._cache})
        if missing:
            for t, v in zip(missing, self._encode(missing)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts], axis=0)


def make_featurizer(cfg):
    """Featurizer factory (the swap point). cfg['model']['text_features']:
    'hashing' (default, offline) or 'stransformer' (real embeddings; the
    bridge's lang side must then use the featurizer's dim, not lang_dim)."""
    m = cfg["model"]
    backend = m.get("text_features", "hashing")
    if backend == "stransformer":
        return SentenceTransformerFeaturizer(
            model_name=m.get("st_model", "sentence-transformers/paraphrase-MiniLM-L3-v2"))
    if backend == "hashing":
        return HashingTextFeaturizer(dim=m["lang_dim"], seed=cfg["seed"],
                                     ngram=m.get("text_ngram", 1))
    raise ValueError(f"unknown text_features backend: {backend}")
