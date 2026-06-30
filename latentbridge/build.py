"""Model factory — builds legacy or v2 modules from config.

Used by training AND by checkpoint loading, so a saved "self" (the learned
knowledge about the user) can be rebuilt identically in a new body (env/robot).

Set ``model.version: v2`` in your config to use the new architecture
(residual blocks, attention world model, cross-attention bridge,
learned text embeddings).
"""
from __future__ import annotations


def make_models(cfg, obs_dim, n_actions, device="cpu"):
    version = cfg.get("model", {}).get("version", "legacy")

    if version == "v2":
        return _make_v2(cfg, obs_dim, n_actions, device)
    else:
        return _make_legacy(cfg, obs_dim, n_actions, device)


def _make_legacy(cfg, obs_dim, n_actions, device="cpu"):
    from .models import Encoder, Decoder, WorldModel, LatentBridge
    from .text_features import HashingTextFeaturizer

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


def _make_v2(cfg, obs_dim, n_actions, device="cpu"):
    from .models_v2 import AttentionWorldModel
    # V2-lite: only upgrade the world model to use self-attention.
    # Residual encoder collapses latent via too-easy alignment.
    from .models import Encoder, Decoder, LatentBridge
    from .text_features import HashingTextFeaturizer

    ld = cfg["model"]["latent_dim"]
    lg = cfg["model"]["lang_dim"]
    h = cfg["model"].get("hidden", 256)
    num_heads = cfg["model"].get("num_heads", 4)
    dropout = cfg["model"].get("dropout", 0.1)

    return {
        "encoder": Encoder(obs_dim, ld, h).to(device),
        "decoder": Decoder(ld, obs_dim, h).to(device),
        "world_model": AttentionWorldModel(ld, n_actions, h, num_heads=num_heads, dropout=dropout).to(device),
        "bridge": LatentBridge(ld, lg, h).to(device),
        "featurizer": HashingTextFeaturizer(dim=lg, seed=cfg["seed"]),
    }
