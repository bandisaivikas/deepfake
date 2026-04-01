# losses.py - NT-Xent loss and training utilities

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent Loss  (Section 3.3, Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.

    For a batch of N pairs (z_i, z_j) the loss for a single positive pair is:

        ℓ_{i,j} = -log[ exp(sim(z_i, z_j)/τ) /
                        Σ_{k≠i} exp(sim(z_i, z_k)/τ) ]

    Both directions (i→j and j→i) are averaged.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        z_i, z_j: (N, D) L2-normalised projection vectors
        """
        batch_size = z_i.size(0)
        device     = z_i.device

        # Normalise
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate → (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity matrix → (2N, 2N)
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pair indices: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0,          batch_size,     device=device)
        ])

        loss = F.cross_entropy(sim, labels)
        return loss
