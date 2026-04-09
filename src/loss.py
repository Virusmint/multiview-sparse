import math
from itertools import combinations
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss for multi-view contrastive learning.
    It can use cosine similarity or any custom similarity function.
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
        return (
            self.sim_metric(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / self.temperature
        )

    def forward(self, latents: List[torch.Tensor]) -> torch.Tensor:
        num_views = len(latents)
        batch_size = latents[0].shape[0]
        device = latents[0].device
        dtype = latents[0].dtype

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        pairs = list(combinations(range(num_views), 2))
        labels = torch.arange(batch_size, device=device)

        for i, j in pairs:
            sim_matrix = self._get_similarity_matrix(latents[i], latents[j])
            loss_ij = F.cross_entropy(sim_matrix, labels)
            loss_ji = F.cross_entropy(sim_matrix.T, labels)
            total_loss += (loss_ij + loss_ji) / 2

        return total_loss / max(len(pairs), 1)


class LpAlignEntropyLoss(nn.Module):
    """
    Theorem-motivated alignment + entropy-style objective.

    The loss has two parts:
      1. Alignment: matched samples across views should be close.
      2. Entropy proxy: samples within each view should not collapse together.

    Minimizing the entropy term below encourages pairwise distances within a
    batch to be larger on average, which acts as a practical anti-collapse term.
    """

    def __init__(
        self,
        p: int = 2,
        tau: float = 1.0,
        use_pow: bool = False,
        align_weight: float = 1.0,
        entropy_weight: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.p = p
        self.tau = tau
        self.use_pow = use_pow
        self.align_weight = align_weight
        self.entropy_weight = entropy_weight
        self.eps = eps

    def _pairwise_lp_distances(self, z: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(z, z, p=self.p)
        dist = dist.clamp_min(self.eps)
        if self.use_pow:
            dist = dist.pow(self.p)
        return dist

    def forward(self, view_latents: List[torch.Tensor]) -> torch.Tensor:
        if len(view_latents) < 2:
            raise ValueError('LpAlignEntropyLoss needs at least two views.')

        num_views = len(view_latents)
        batch_size = view_latents[0].shape[0]
        device = view_latents[0].device
        dtype = view_latents[0].dtype

        if batch_size < 2:
            raise ValueError('Batch size must be at least 2 for entropy regularization.')

        # 1) Alignment across corresponding samples from different views.
        align_loss = torch.tensor(0.0, device=device, dtype=dtype)
        pairs = list(combinations(range(num_views), 2))
        for i, j in pairs:
            dist = torch.norm(view_latents[i] - view_latents[j], p=self.p, dim=-1)
            if self.use_pow:
                dist = dist.pow(self.p)
            align_loss += dist.mean()
        align_loss = align_loss / len(pairs)

        # 2) Entropy-style anti-collapse term within each view.
        #    We exclude the diagonal and minimize log-mean-exp(-distance / tau).
        #    If within-view samples spread out, this quantity becomes smaller.
        entropy_term = torch.tensor(0.0, device=device, dtype=dtype)
        diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        log_num_neg = math.log(batch_size - 1)

        for z in view_latents:
            pairwise_dist = self._pairwise_lp_distances(z)
            pairwise_dist = pairwise_dist.masked_fill(diag_mask, float('inf'))
            log_mean_exp = torch.logsumexp(-pairwise_dist / self.tau, dim=1) - log_num_neg
            entropy_term += log_mean_exp.mean()

        entropy_term = entropy_term / num_views
        return self.align_weight * align_loss + self.entropy_weight * entropy_term
