import torch
import torch.nn.functional as F
import torch.nn as nn


class SymmetricedInfoNCE(nn.Module):
    """
    InfoNCE Loss function symmetrized
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    # # cosine sim
    # def forward(self, z1, z2):
    #     # Normalize to unit sphere
    #     z1 = F.normalize(z1, dim=1)
    #     z2 = F.normalize(z2, dim=1)
    #     # Cosine similarity logits
    #     logits = torch.matmul(z1, z2.t()) / self.temperature
    #     # Positive pairs on diagonal
    #     labels = torch.arange(logits.size(0), device=logits.device)
    #     # Symmetrized cross-entropy (F.cross_entropy applies log-softmax)
    #     loss_1 = F.cross_entropy(logits, labels)
    #     loss_2 = F.cross_entropy(logits.t(), labels)
    #     return (loss_1 + loss_2) / 2

    # negative l2 distance
    def forward(self, z1, z2):
        dist_sq = torch.cdist(z1, z2, p=2).pow(2)
        # Convert distance to a "logit" (Similarity)
        # We negate it because smaller distance must mean higher probability
        logits = -dist_sq / self.temperature
        # Standard InfoNCE logic
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_1 = F.cross_entropy(logits, labels)
        loss_2 = F.cross_entropy(logits.t(), labels)
        return (loss_1 + loss_2) / 2
