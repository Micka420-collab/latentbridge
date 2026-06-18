"""Model factory — one place that builds the modules from a config.

Used by training AND by checkpoint loading, so a saved "self" (the learned
knowledge about the user) can be rebuilt identically in a new body (env/robot).
"""
from __future__ import annotations

from .models import Encoder, Decoder, WorldModel, LatentBridge
from .text_features import HashingTextFeaturizer


def make_models(cfg, obs_dim, n_actions, device="cpu"):
    ld = cfg["model"]["latent_dim"]
    lg = cfg["model"]["lang_dim"]
    h = cfg["model"]["hidden"]
    return {
        "encoder": Encoder(obs_dim, ld, h).to(device),
        "decoder": Decoder(ld, obs_dim, h).to(device),
        "world_model": WorldModel(ld, n_actions, h).to(device),
        "bridge": LatentBridge(ld, lg, h).to(device),
        "featurizer": HashingTextFeaturizer(dim=lg, seed=cfg["seed"]),
    }
