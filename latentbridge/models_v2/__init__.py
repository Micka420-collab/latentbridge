"""LatentBridge v2 — residual + attention architecture.

Exports:
  ResidualEncoder   — encoder with skip connections + LayerNorm
  ResidualDecoder   — decoder with skip connections + LayerNorm
  AttentionWorldModel — world model with multi-head self-attention
  CrossAttBridge     — bridge with cross-attention + contrastive loss
"""

from .residual_encoder import ResidualEncoder
from .residual_decoder import ResidualDecoder
from .attention_world_model import AttentionWorldModel
from .cross_att_bridge import CrossAttBridge

__all__ = [
    "ResidualEncoder",
    "ResidualDecoder",
    "AttentionWorldModel",
    "CrossAttBridge",
]
