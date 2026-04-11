import torch
import torch.nn.functional as F


def dot_product(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return torch.mm(z1, z2.T)


def neg_l2_dist(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return -torch.cdist(z1, z2)


def cosine_sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1_norm = F.normalize(z1, p=2, dim=1)
    z2_norm = F.normalize(z2, p=2, dim=1)
    return torch.mm(z1_norm, z2_norm.T)
