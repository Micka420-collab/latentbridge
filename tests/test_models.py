"""Shape / gradient / determinism tests for the learned modules."""
import numpy as np
import torch

from latentbridge.models import (Decoder, Encoder, LatentBridge,
                                 RecurrentWorldModel, WorldModel)
from latentbridge.text_features import HashingTextFeaturizer

B, OBS, LAT, LANG, ACT = 5, 18, 8, 12, 4


def test_encoder_decoder_shapes():
    enc, dec = Encoder(OBS, LAT, 16), Decoder(LAT, OBS, 16)
    z = enc(torch.randn(B, OBS))
    assert z.shape == (B, LAT)
    assert dec(z).shape == (B, OBS)


def test_world_model_forward_and_rollout():
    wm = WorldModel(LAT, ACT, 16)
    z = torch.randn(B, LAT)
    a = torch.randint(0, ACT, (B,))
    zn, r = wm(z, a)
    assert zn.shape == (B, LAT) and r.shape == (B,)

    H = 6
    zs, rs = wm.rollout(z, torch.randint(0, ACT, (B, H)))
    assert zs.shape == (B, H, LAT) and rs.shape == (B, H)


def test_recurrent_world_model_carries_state():
    wm = RecurrentWorldModel(LAT, ACT, 16, det_dim=10)
    z = torch.randn(B, LAT)
    a = torch.randint(0, ACT, (B,))
    zn, r, g = wm(z, a)          # fresh state created internally
    assert zn.shape == (B, LAT) and r.shape == (B,) and g.shape == (B, 10)
    # the recurrent state must influence the next prediction
    zn2, _, _ = wm(z, a, g)
    zn_fresh, _, _ = wm(z, a)
    assert not torch.allclose(zn2, zn_fresh)


def test_bridge_losses_finite_and_trainable():
    bridge = LatentBridge(LAT, LANG, 16)
    z = torch.randn(B, LAT)
    tgt = torch.randn(B, LANG)
    loss = bridge.align_loss(z, tgt) + bridge.cycle_loss(z)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in bridge.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_featurizer_deterministic_unit_norm_and_cached():
    f = HashingTextFeaturizer(dim=32, seed=0)
    v1 = f.embed("agent at row 1 col 2")
    v2 = f.embed("agent at row 1 col 2")
    assert np.array_equal(v1, v2)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5
    # different seed -> different space
    v3 = HashingTextFeaturizer(dim=32, seed=1).embed("agent at row 1 col 2")
    assert not np.array_equal(v1, v3)
    # empty text -> zero vector, no crash
    assert np.array_equal(f.embed("!!!"), np.zeros(32, np.float32))
    # batch matches single
    batch = f.embed_batch(["agent at row 1 col 2", "goal reached"])
    assert np.array_equal(batch[0], v1)
