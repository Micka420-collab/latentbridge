"""Cross-Attention Bridge between latent space and language space.

Replaces the simple MLP bridge with cross-attention:
- latent query attends to language keys/values
- language query attends to latent keys/values

Plus a contrastive loss (InfoNCE-style) for much stronger alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttBridge(nn.Module):
    """Bidirectional bridge with cross-attention.

    to_lang(z):  z projects to lang space, then cross-attends with a learned
                 language "template" to produce the final lang embedding.
    from_lang(e): e projects to latent space, then cross-attends with a learned
                   latent "template" to produce the final latent.

    The cross-attention lets the model learn *which parts* of the latent
    matter for each language dimension, and vice versa — vastly more
    expressive than a fixed MLP projection.
    """

    def __init__(self, latent_dim: int, lang_dim: int, hidden: int = 256,
                 num_queries: int = 8, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.lang_dim = lang_dim
        self.hidden = hidden

        # Learnable query tokens for cross-attention
        self.lang_queries = nn.Parameter(torch.randn(num_queries, hidden))
        self.latent_queries = nn.Parameter(torch.randn(num_queries, hidden))

        # Projections
        self.z_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.e_to_hidden = nn.Sequential(
            nn.Linear(lang_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )

        # Cross-attention layers
        self.to_lang_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.from_lang_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        # Output projections
        self.to_lang_out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, lang_dim),
        )
        self.from_lang_out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def to_lang(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent z to language space via cross-attention."""
        B = z.shape[0]
        # z acts as key/value, learned lang_queries act as query
        k_v = self.z_to_hidden(z).unsqueeze(1)  # (B, 1, hidden)
        # Expand latent to provide richer keys
        k_v = k_v.repeat(1, 4, 1)  # (B, 4, hidden) — give it some multiplicity
        q = self.lang_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, hidden)
        attn_out, _ = self.to_lang_attn(q, k_v, k_v)
        # Pool queries
        pooled = attn_out.mean(dim=1)  # (B, hidden)
        return self.to_lang_out(pooled)

    def from_lang(self, e: torch.Tensor) -> torch.Tensor:
        """Project language embedding to latent space via cross-attention."""
        B = e.shape[0]
        k_v = self.e_to_hidden(e).unsqueeze(1).repeat(1, 4, 1)  # (B, 4, hidden)
        q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, hidden)
        attn_out, _ = self.from_lang_attn(q, k_v, k_v)
        pooled = attn_out.mean(dim=1)  # (B, hidden)
        return self.from_lang_out(pooled)

    def align_loss(self, z: torch.Tensor, lang_target: torch.Tensor) -> torch.Tensor:
        """Cosine alignment between projected latent and language target."""
        proj = self.to_lang(z)
        proj = F.normalize(proj, dim=-1)
        tgt = F.normalize(lang_target, dim=-1)
        return (1.0 - (proj * tgt).sum(-1)).mean()

    def contrastive_loss(self, z: torch.Tensor, lang_target: torch.Tensor,
                         temperature: float = 0.07) -> torch.Tensor:
        """InfoNCE contrastive loss: pulls matching (z, lang) pairs together,
        pushes non-matching pairs apart in the shared embedding space.

        This is MUCH stronger than simple cosine alignment — it creates a
        proper embedding space where latents and language are clustered
        by semantics.
        """
        z_proj = F.normalize(self.to_lang(z), dim=-1)
        t_proj = F.normalize(lang_target, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(z_proj, t_proj.T) / temperature  # (B, B)

        # Labels: diagonal is the positive pair
        labels = torch.arange(z.shape[0], device=z.device)

        # Symmetric loss
        loss_z = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_z + loss_t) / 2

    def cycle_loss(self, z: torch.Tensor) -> torch.Tensor:
        """from_lang(to_lang(z)) should reconstruct z."""
        return F.mse_loss(self.from_lang(self.to_lang(z)), z)
