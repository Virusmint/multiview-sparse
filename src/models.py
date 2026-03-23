import torch
import torch.nn as nn
from typing import List


class Encoder(nn.Module):
    """
    Base encoder architecture for views.
    """

    def __init__(self, layers: List[int]) -> None:
        super().__init__()
        # Build modules
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.LeakyReLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class BimodalEncoders(nn.Module):
    def __init__(self, layers1: List[int], layers2: List[int], criterion, lr=1e-4):
        super().__init__()
        self.enc1 = Encoder(layers1)
        self.enc2 = Encoder(layers2)
        self.criterion = criterion
        self.lr = lr

    def forward(self, x1, x2):
        return self.enc1(x1), self.enc2(x2)

    def training_step(self, batch):
        x1, x2, _ = batch
        z1_hat, z2_hat = self(x1, x2)
        loss = self.criterion(z1_hat, z2_hat)
        return loss

    def validation_step(self, batch):
        x1, x2, _ = batch
        z1_hat, z2_hat = self(x1, x2)
        loss = self.criterion(z1_hat, z2_hat)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
