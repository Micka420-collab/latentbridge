"""Learned text embedder — replaces HashingTextFeaturizer.

Instead of a fixed hashing trick, this learns a small embedding table
that gets fine-tuned during joint training. Tokens are hashed to indices
(like the old featurizer) but the embeddings themselves are learnable
parameters, so the model can adapt the language representation to the
task.

Architecture:
  tokenize text -> hash to indices -> learned Embedding lookup -> mean pool -> output
"""

import hashlib
import re

import numpy as np
import torch
import torch.nn as nn


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer. No heavy NLP deps."""
    return re.findall(r"[a-zA-Z0-9]+|[^\s]", text.lower())


class LearnedTextEmbedder(nn.Module):
    """Learnable text embedding table with hashed token indices."""

    def __init__(self, dim: int = 128, vocab_size: int = 4096, seed: int = 42):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)

        # Learnable embedding table
        self.embed = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def _hash_token(self, token: str) -> int:
        """Hash a token to a vocab index (deterministic)."""
        h = hashlib.md5(token.encode()).digest()
        return int.from_bytes(h[:4], "little") % self.vocab_size

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string -> numpy array (used during data collection)."""
        tokens = _tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        indices = [self._hash_token(t) for t in tokens]
        with torch.no_grad():
            idx_tensor = torch.as_tensor(indices)
            embeds = self.embed(idx_tensor)
            return embeds.mean(dim=0).cpu().numpy()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts -> (N, dim) numpy array."""
        return np.stack([self.embed_text(t) for t in texts])

    def forward(self, texts: list[str], device: str = "cpu") -> torch.Tensor:
        """Embed a batch and return a torch tensor (differentiable!)."""
        all_indices = []
        max_len = 0
        for text in texts:
            tokens = _tokenize(text)
            if not tokens:
                tokens = ["."]  # fallback for empty
            indices = [self._hash_token(t) for t in tokens]
            all_indices.append(indices)
            max_len = max(max_len, len(indices))

        # Pad to max_len
        padded = []
        mask = []
        for indices in all_indices:
            pad_len = max_len - len(indices)
            padded.append(indices + [0] * pad_len)
            mask.append([1.0] * len(indices) + [0.0] * pad_len)

        idx_tensor = torch.as_tensor(padded, device=device)
        mask_tensor = torch.as_tensor(mask, device=device).unsqueeze(-1)

        embeds = self.embed(idx_tensor)  # (B, L, dim)
        # Masked mean pool
        pooled = (embeds * mask_tensor).sum(dim=1) / mask_tensor.sum(dim=1).clamp(min=1)
        return pooled
