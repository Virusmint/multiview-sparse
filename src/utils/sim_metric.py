import torch
import torch.nn.functional as F


def normalized_neg_l2_sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    dist_sq = 2 - 2 * torch.mm(z1, z2.T)
    return -dist_sq


def cosine_sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1_norm = F.normalize(z1, p=2, dim=1)
    z2_norm = F.normalize(z2, p=2, dim=1)
    return torch.mm(z1_norm, z2_norm.T)
