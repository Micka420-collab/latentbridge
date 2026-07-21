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
    def __init__(self, dim: int = 128, seed: int = 0):
        self.dim = int(dim)
        self.seed = int(seed)
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
        v = np.mean([self._token_vec(t) for t in toks], axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_batch(self, texts) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts], axis=0)
