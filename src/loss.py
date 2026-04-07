import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Callable


class SymInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Loss function for multi-view contrastive learning.
    Can be used with cosine similarity or negative L2 distance as the similarity measure.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        sim_metric: Callable[..., torch.Tensor] = F.cosine_similarity,
    ):
        super().__init__()
        self.temperature = temperature
        self.sim_metric = sim_metric

    def _get_similarity_matrix(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the similarity matrix between two batches of latents."""
        return (
            self.sim_metric(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / self.temperature
        )

    def forward(self, latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the symmetric contrastive loss across all pairs of views.
        """
        num_views = len(latents)
        batch_size = latents[0].shape[0]
        device = latents[0].device
        dtype = latents[0].dtype

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        pair_count = num_views * (num_views - 1) / 2  # Number of unique pairs

        # Create labels (diagonal matrix) once
        labels = torch.arange(batch_size).to(device)

        # Iterate through all unique pairs of views
        for i in range(num_views):
            for j in range(i + 1, num_views):
                sim_matrix = self._get_similarity_matrix(latents[i], latents[j])

                # Symmetric cross-entropy (View i -> j and View j -> i)
                loss_ij = F.cross_entropy(sim_matrix, labels)
                loss_ji = F.cross_entropy(sim_matrix.T, labels)

                total_loss += (loss_ij + loss_ji) / 2

        return total_loss / pair_count


class LpAlignEntropyLoss(nn.Module):
    """
    Special case of the general InfoNCE loss, where:
        - tau = 1.0 (no temperature scaling)
        - sim_metric = negative Lp distance
        - K -> infinity (only the closest negative sample contributes to the loss)
    Implements Theorem 3.2: Content Alignment + Entropy Regularization.
    """

    def __init__(self, p: int = 2, tau: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.p = p
        self.tau = tau  # Temperature
        self.alpha = alpha  # Weighting between alignment and entropy

    def forward(self, view_latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            view_latents: List of Tensors [z1, z2, ..., zV] each of shape (B, D)
        """
        num_views = len(view_latents)
        device = view_latents[0].device
        dtype = view_latents[0].dtype

        pos_loss = torch.tensor(0.0, device=device, dtype=dtype)
        neg_loss = torch.tensor(0.0, device=device, dtype=dtype)
        pair_count = num_views * (num_views - 1) / 2  # Number of unique pairs

        # Iterate through all unique pairs of views
        for i in range(num_views):
            for j in range(i + 1, num_views):
                # 1. Content Alignment: distance between the same sample in different views
                dist_pos = torch.norm(
                    view_latents[i] - view_latents[j], p=self.p, dim=-1
                )  # (B,)
                pos_loss += dist_pos.mean()

                # 2. Entropy Regularization: approximate H(z) by pushing samples apart in the latent space
                dist_matrix = torch.cdist(view_latents[i], view_latents[j], p=self.p)
                neg_loss += self._logmeanexp(
                    -dist_matrix / self.tau, dim=1
                ).mean()  # Minimize log(mean(exp(-d))) to maximize entropy

        # Combine terms based on the alpha weight
        total_loss = (self.alpha * pos_loss + (1 - self.alpha) * neg_loss) / pair_count
        return total_loss

    def _logmeanexp(self, x, dim):
        # Numerically stable log(mean(exp(x)))
        return torch.logsumexp(x, dim=dim) - torch.log(
            torch.tensor(x.shape[dim], dtype=x.dtype, device=x.device)
        )
