import torch
import torch.nn as nn
from typing import List


class MultiViewEncoders(nn.Module):
    """
    Encodes multiple views into a shared latent space.
    """

    # NOTE: Each encoder has the same architecture, except for the input dimension. We could modify this to allow for different architectures per view if needed.
    def __init__(self, view_encoders: List[nn.Module]) -> None:
        super().__init__()
        # Create a list of view-specific encoders
        self.encoders = nn.ModuleList(view_encoders)

    def forward(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        # Encodes each view into latent representation
        latents = [enc(x) for enc, x in zip(self.encoders, views)]
        return latents


class MLPEncoder(nn.Module):
    """
    MLP encoder architecture for views.
    """

    def __init__(self, input_dim, hidden_dims: List[int], output_dim) -> None:
        super().__init__()
        layers = [input_dim] + hidden_dims + [output_dim]
        # Build modules
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.LeakyReLU(0.2))
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
