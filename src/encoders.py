import torch
import torch.nn as nn
from typing import List
from src.hard_concrete import HardConcreteGate


class MultiViewEncoders(nn.Module):
    """
    Encodes multiple views into a shared latent space.

    If use_sparsity is True, applies a shared hard concrete gate to encourage learning a common subset of latent factors across views.
    Otherwise, the true content size is assumed to be given by the output dimension of the encoders.
    """

    # NOTE: Each encoder has the same architecture, except for the input dimension. We could modify this to allow for different architectures per view if needed.
    def __init__(self, view_encoders: List, use_sparsity: bool = False):
        super().__init__()
        # Create a list of view-specific encoders
        self.encoders = nn.ModuleList(view_encoders)
        self.use_sparsity = use_sparsity
        if use_sparsity:
            # Single gate shared across all views to encourage learning a common subset of latent factors
            self.gates = HardConcreteGate(dim=view_encoders[0].net[-1].out_features)
        else:
            self.gates = nn.Identity()  # No gating if not using sparsity

    def forward(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        # Encodes each view into latent representation
        latents = [enc(x) for enc, x in zip(self.encoders, views)]
        sparse_latents = [self.gates(latent) for latent in latents]
        return sparse_latents

    def get_l0_penalty(self) -> torch.Tensor:
        if self.use_sparsity:
            return self.gates.get_l0_penalty()  # type: ignore[attr-defined]
        return torch.tensor(
            0.0, device=next(self.parameters()).device
        )  # No penalty if not using sparsity

    def get_active_indices(self) -> List[int]:
        """
        Returns the indices of the active latent factors based on the hard concrete gates.
        If not using sparsity, returns all factors as active.
        """
        if self.use_sparsity:
            return self.gates.get_active_indices()  # type: ignore[attr-defined]
        return list(
            range(self.encoders[0].net[-1].out_features)
        )  # All factors are active


class MLPEncoder(nn.Module):
    """
    MLP encoder architecture for views.
    """

    def __init__(self, input_dim, hidden_dims: List[int], output_dim) -> None:
        assert len(hidden_dims) > 0, "Must specify at least one hidden layer"
        super().__init__()
        layers = [input_dim] + hidden_dims + [output_dim]

        # Build modules
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.LeakyReLU(0.2))
            # if i == len(layers) - 2:
            #     modules.append(nn.Sigmoid())  # Final activation for latent space

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class ConvEncoder(nn.Module):
    """
    Convolutional encoder architecture for images.
    """

    def __init__(self, input_channels, hidden_dims: List[int], output_dim) -> None:
        super().__init__()
        layers = []
        in_channels = input_channels
        for hidden_dim in hidden_dims:
            layers.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
            )
            layers.append(nn.LeakyReLU(0.2))
            in_channels = hidden_dim
        self.conv_net = nn.Sequential(*layers)
        self.fc = nn.Linear(
            in_channels * 8 * 8, output_dim
        )  # Assuming input images are 32x32

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)
