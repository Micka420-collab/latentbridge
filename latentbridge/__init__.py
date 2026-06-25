"""LatentBridge: a hybrid LLM + world-model architecture.

The core idea ("Latent Bridge"): a learned world model and a frozen LLM share a
learned latent interface. Three jointly-trained objectives:
  1. world-model dynamics  (predict next latent + reward, self-supervised)
  2. reconstruction        (ground the latent in observations)
  3. latent<->language alignment  (the novel piece: the adapter makes the
     world-model latent space speak the LLM's language)

The gridworld is the proving ground. The same architecture is meant to later
point at a "personal world model" (the user's own context) for a life assistant.
"""

__version__ = "0.1.0"
