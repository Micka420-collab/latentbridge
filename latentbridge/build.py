"""Model factory — one place that builds the modules from a config.

Used by training AND by checkpoint loading, so a saved "self" (the learned
knowledge about the user) can be rebuilt identically in a new body (env/robot).
"""
from __future__ import annotations

from .models import Encoder, Decoder, WorldModel, RecurrentWorldModel, LatentBridge
from .text_features import make_featurizer


def make_models(cfg, obs_dim, n_actions, device="cpu"):
    ld = cfg["model"]["latent_dim"]
    h = cfg["model"]["hidden"]
    # the featurizer defines the language space; the bridge's lang side must
    # match ITS dim (== model.lang_dim for hashing, the ST model's dim otherwise)
    feat = make_featurizer(cfg)
    img = cfg["model"].get("obs_image_shape")   # [C,H,W] -> use a CNN (vision)
    if img:
        from .models.vision import ConvEncoder, ConvDecoder
        enc = ConvEncoder(tuple(img), ld, h).to(device)
        dec = ConvDecoder(ld, tuple(img), h).to(device)
    else:
        enc = Encoder(obs_dim, ld, h).to(device)
        dec = Decoder(ld, obs_dim, h).to(device)
    if cfg["model"].get("dynamics") == "rssm":
        wm = RecurrentWorldModel(ld, n_actions, h,
                                 det_dim=cfg["model"].get("det_dim")).to(device)
    else:
        wm = WorldModel(ld, n_actions, h).to(device)
    return {
        "encoder": enc,
        "decoder": dec,
        "world_model": wm,
        "bridge": LatentBridge(ld, feat.dim, h).to(device),
        "featurizer": feat,
    }
