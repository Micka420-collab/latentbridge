import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentBridge(nn.Module):
    """The novel piece: a learned, bidirectional adapter between the world
    model's latent space and the LLM's language/embedding space.

    - to_lang(z):  project a world latent into language space, so an LLM (or a
      language-embedding target) can "read" the agent's internal state.
    - from_lang(e): project a language embedding (e.g. an LLM-proposed subgoal)
      back into latent space, so the world model can plan toward it.

    Trained by an alignment loss that pulls to_lang(z) toward the language
    embedding of the ground-truth state description. This is what makes the
    latent space *linguistically grounded* rather than an opaque blob.
    """

    def __init__(self, latent_dim: int, lang_dim: int, hidden: int = 128):
        super().__init__()
        self.to_lang = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, lang_dim),
        )
        self.from_lang = nn.Sequential(
            nn.Linear(lang_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def align_loss(self, z: torch.Tensor, lang_target: torch.Tensor) -> torch.Tensor:
        """Cosine alignment between projected latent and the language target."""
        proj = self.to_lang(z)
        proj = F.normalize(proj, dim=-1)
        tgt = F.normalize(lang_target, dim=-1)
        return (1.0 - (proj * tgt).sum(-1)).mean()

    def cycle_loss(self, z: torch.Tensor) -> torch.Tensor:
        """from_lang(to_lang(z)) should reconstruct z (keeps the bridge invertible)."""
        return F.mse_loss(self.from_lang(self.to_lang(z)), z)
