import torch
import torch.nn as nn
from torchvision.models import resnet18
from src.hard_concrete import HardConcreteGate
from typing import List, Sequence
from abc import ABC, abstractmethod


class ViewEncoder(nn.Module, ABC):
    """
    Base abstract class for view encoder. Can be extended to implement specific architectures for different modalities.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MultiViewEncoders(nn.Module):
    """
    Encodes multiple views into a shared latent space.

    Applies a shared hard concrete gate to encourage learning a common subset of latent factors across views.
    Otherwise, the true content size is assumed to be given by the output dimension of the encoders.
    """

    # NOTE: Each encoder has the same architecture, except for the input dimension.
    # We could modify this to allow for different architectures per view if needed.
    def __init__(self, view_encoders: Sequence[ViewEncoder]):
        super().__init__()
        # Create a list of view-specific encoders
        self.encoders = nn.ModuleList(view_encoders)
        self.output_dim = self._get_encoder_output_dim(self.encoders[0])
        self.gate = HardConcreteGate(dim=self.output_dim)

    def forward(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        # Encodes each view into latent representation
        latents = [enc(x) for enc, x in zip(self.encoders, views)]
        sparse_latents = [self.gate(latent) for latent in latents]
        return sparse_latents

    def get_l0_penalty(self) -> torch.Tensor:
        """Returns the current L0 penalty from the hard concrete gate, which can be used for regularization during training."""
        return self.gate.get_l0_penalty()

    def get_gate_values(self) -> torch.Tensor:
        """Returns the current values of the hard concrete gate (deterministic mask)"""
        return self.gate.get_values()

    @staticmethod
    def _get_encoder_output_dim(encoder: nn.Module) -> int:
        """
        Infers the output dimension of the encoder module by checking common attribute patterns.
        Assumes the encoder has a linear layer at the end.
        """
        if hasattr(encoder, "net") and isinstance(encoder.net, nn.Sequential):
            for module in reversed(encoder.net):
                if hasattr(module, "out_features"):
                    return int(module.out_features)
        if hasattr(encoder, "linear") and hasattr(encoder.linear, "out_features"):
            return int(encoder.linear.out_features)
        if hasattr(encoder, "fc") and hasattr(encoder.fc, "out_features"):
            return int(encoder.fc.out_features)
        raise ValueError(
            "Could not infer encoder output dimension. "
            "Expected one of: encoder.net[-1].out_features, encoder.linear.out_features, or encoder.fc.out_features."
        )


class MLPEncoder(ViewEncoder):
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
            # Use LeakyReLU for all layers except the last one
            if i < len(layers) - 2:
                modules.append(nn.LeakyReLU(0.2))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class ImageEncoderResNet(ViewEncoder):
    """
    Reference-style image encoder:
    ResNet18 -> LeakyReLU -> Linear(out_dim).
    """

    def __init__(self, output_dim: int, hidden_dim: int = 100) -> None:
        super().__init__()
        self.net = nn.Sequential(
            resnet18(num_classes=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextEncoder2D(ViewEncoder):
    """
    Reference-style 2D ConvNet text encoder used for multimodal image/text experiments.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sequence_length: int,
        embedding_dim: int = 128,
        fbase: int = 25,
    ):
        super().__init__()
        if sequence_length < 24 or sequence_length > 31:
            raise ValueError("TextEncoder2D expects sequence_length between 24 and 31")
        self.fbase = fbase
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.convnet = nn.Sequential(
            nn.Conv2d(1, fbase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase),
            nn.ReLU(True),
            nn.Conv2d(fbase, fbase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase * 2),
            nn.ReLU(True),
            nn.Conv2d(fbase * 2, fbase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase * 4),
            nn.ReLU(True),
        )
        self.ldim = fbase * 4 * 3 * 16
        self.linear = nn.Linear(self.ldim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).unsqueeze(1)
        x = self.convnet(x)
        x = x.view(-1, self.ldim)
        return self.linear(x)
