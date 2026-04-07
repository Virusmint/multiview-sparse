import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Callable
from itertools import combinations


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
        pairs = list(combinations(range(num_views), 2))  # Unique pairs of views

        # Create labels (diagonal matrix) once
        labels = torch.arange(batch_size).to(device)

        # Iterate through all unique pairs of views
        for i, j in pairs:
            sim_matrix = self._get_similarity_matrix(latents[i], latents[j])

            # Symmetric cross-entropy (View i -> j and View j -> i)
            loss_ij = F.cross_entropy(sim_matrix, labels)
            loss_ji = F.cross_entropy(sim_matrix.T, labels)

            total_loss += (loss_ij + loss_ji) / 2

        return total_loss


# BUG: Doesn't work
class LpAlignEntropyLoss(nn.Module):
    """
    Special case of the general InfoNCE loss, where:
        - tau = 1.0 (no temperature scaling)
        - sim_metric = negative Lp distance
        - K -> infinity (only the closest negative sample contributes to the loss)
    Implements Theorem 3.2: Content Alignment + Entropy Regularization.
    """

    def __init__(self, p: int = 2, tau: float = 1.0, use_pow: bool = False, eps=1e-8):
        super().__init__()
        self.p = p
        self.tau = tau  # Temperature
        self.use_pow = use_pow  # Use p-th power in distance calculations
        self.eps = eps  # Numerical stability

    def forward(self, view_latents: List[torch.Tensor]) -> torch.Tensor:
        num_views = len(view_latents)
        batch_size = view_latents[0].shape[0]
        device = view_latents[0].device
        dtype = view_latents[0].dtype

        # 1. Content Alignment (p-th power)
        align_loss = torch.tensor(0.0, device=device, dtype=dtype)
        pairs = list(combinations(range(num_views), 2))  # Unique pairs of views

        for i, j in pairs:
            # Distance between corresponding samples in view i and view j
            dist = torch.norm(
                view_latents[i] - view_latents[j] + self.eps, p=self.p, dim=-1
            )
            if self.use_pow:
                dist = dist.pow(self.p)
            align_loss += dist.mean()

        align_loss /= len(pairs)

        # 2. Entropy Regularization
        entropy_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for z in view_latents:
            # Compute pairwise distance matrix for each view
            dist = torch.norm(
                z.unsqueeze(1) - z.unsqueeze(0) + self.eps, p=self.p, dim=-1
            )
            if self.use_pow:
                dist = dist.pow(self.p)
            # Exclude self-similarity by masking the diagonal
            mask = torch.eye(batch_size, device=device, dtype=torch.bool)
            neg_samples = dist[~mask].view(
                batch_size, -1
            )  # Shape: (batch_size, batch_size-1)
            log_mean_exp = torch.logsumexp(-neg_samples / self.tau, dim=1) - torch.log(
                torch.tensor(batch_size - 1, dtype=dtype, device=device)
            )
            entropy_loss += log_mean_exp.mean()

        entropy_loss /= num_views

        # print(
        #     f"Alignment Loss: {align_loss.item():.4f}, Entropy Loss: {entropy_loss.item():.4f}"
        # )

        return align_loss - entropy_loss
