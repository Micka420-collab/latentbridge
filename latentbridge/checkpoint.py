"""Save / load the learned "self".

A checkpoint is the portable memory: the world model + the latent<->language
bridge that encode what the system has learned. Transferring the assistant into
another body (a bigger world, eventually a robot) = loading this checkpoint and
attaching a new env. The bridge keeps the same language interface across bodies.
"""
from __future__ import annotations

import torch

from .build import make_models


def save(path, modules, cfg, obs_dim, n_actions):
    torch.save({
        "cfg": cfg,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "encoder": modules["encoder"].state_dict(),
        "decoder": modules["decoder"].state_dict(),
        "world_model": modules["world_model"].state_dict(),
        "bridge": modules["bridge"].state_dict(),
    }, path)


def load(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = make_models(ckpt["cfg"], ckpt["obs_dim"], ckpt["n_actions"], device)
    m["encoder"].load_state_dict(ckpt["encoder"])
    m["decoder"].load_state_dict(ckpt["decoder"])
    m["world_model"].load_state_dict(ckpt["world_model"])
    m["bridge"].load_state_dict(ckpt["bridge"])
    for k in ("encoder", "decoder", "world_model", "bridge"):
        m[k].eval()
    m["history"] = []
    return m, ckpt["cfg"], ckpt["obs_dim"], ckpt["n_actions"]
