import torch
import torch.nn as nn
from typing import List

# -- ONLY FOR NUMERICAL EXPERIMENTS --
# Mixing functions for the multi-view data-generating process.


class MixingFunction(nn.Module):
    """
    Invertible MLP to use as mixing function for the data-generating process.
    """

    @staticmethod
    def init_orthogonal(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)  # For full rank matrices
            nn.init.zeros_(module.bias)

    def __init__(self, input_dim, n_layers: int = 2):
        super().__init__()
        modules = [nn.Linear(input_dim, input_dim), nn.LeakyReLU(0.1)] * n_layers
        self.net = nn.Sequential(*modules)
        self.net.apply(self.init_orthogonal)

        # Freeze model weights (mixing function is taken as ground truth)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, z):
        return self.net(z)


class MultiViewMixer(nn.Module):
    """
    A multi-view mixing function that applies separate mixing functions to each view.
    """

    def __init__(self, view_configs: List[List[int]]):
        """
        view_configs: List of lists containing latent indices for each view.
        Example: [[0, 1], [0, 2]] means view 1 gets latent factors 0 and 1, and view 2 gets latent factors 0 and 2.
        """
        super().__init__()
        self.view_configs = view_configs
        self.mixers = nn.ModuleList()
        for view_indices in view_configs:
            input_dim = len(view_indices)
            self.mixers.append(MixingFunction(input_dim))

    def forward(self, z_global: torch.Tensor) -> List[torch.Tensor]:
        # Extract the relevant latent factors for each view
        observations = []
        for i, indices in enumerate(self.view_configs):
            z_view = z_global[:, indices]  # Extract latent factors for this view
            x_view = self.mixers[i](z_view)  # Apply the mixing function for this view
            observations.append(x_view)
        return observations


def generate_covariance(dim):
    """
    Generate a random symmetric positive-definite matrix of size dim to be used as covariance matrix
    """
    A = torch.randn(dim, dim)
    return torch.mm(A, A.t()) + torch.eye(dim) * 1e-3
