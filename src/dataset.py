import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def generate_covariance(dim):
    """
    Generate a random symmetric positive-definite matrix of size dim to be used as covariance matrix
    """
    A = torch.randn(dim, dim)
    return torch.mm(A, A.t()) + torch.eye(dim) * 1e-3


class MixingFunction(nn.Module):
    """
    Nonlinear invertible MLP to use as mixing function for the data-generating process.
    """

    @staticmethod
    def init_orthogonal(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)  # For full rank matrices
            nn.init.zeros_(module.bias)

    def __init__(self, dim, n_layers):
        super().__init__()
        modules = [nn.Linear(dim, dim), nn.LeakyReLU(0.1)] * n_layers
        self.net = nn.Sequential(*modules)
        self.net.apply(self.init_orthogonal)

        # Freeze model weights (mixing function is taken as ground truth)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, z):
        return self.net(z)


class NumericalDataset:
    def __init__(
        self,
        num_samples=20000,
        dim_c=4,
        dim_s=4,
        dim_m=2,
        causal=True,
        pi=0.5,
        batch_size=64,
        train_test_split=0.8,
    ):
        self.batch_size = batch_size
        self.num_train = int(num_samples * train_test_split)
        self.num_val = num_samples - self.num_train

        self.dim_total = dim_c + dim_s + dim_m
        # Sample Base Variables
        self.c = torch.distributions.MultivariateNormal(
            torch.zeros(dim_c), generate_covariance(dim_c)
        ).sample((num_samples,))
        self.m1 = torch.distributions.MultivariateNormal(
            torch.zeros(dim_m), generate_covariance(dim_m)
        ).sample((num_samples,))
        self.m2 = torch.distributions.MultivariateNormal(
            torch.zeros(dim_m), generate_covariance(dim_m)
        ).sample((num_samples,))

        # Induce Causal Dependence (c -> s)
        if causal:
            a = torch.randn(dim_s)
            B = torch.randn(dim_s, dim_c)
            mu_s = a + torch.matmul(self.c, B.t())
        else:
            mu_s = torch.zeros(num_samples, dim_s)

        self.s = mu_s + torch.distributions.MultivariateNormal(
            torch.zeros(dim_s), generate_covariance(dim_s)
        ).sample((num_samples,))

        # Style Perturbation for View 2 (Add noise to each style dimension independently with probability pi)
        eps = torch.distributions.MultivariateNormal(
            torch.zeros(dim_s), generate_covariance(dim_s)
        ).sample((num_samples,))
        mask = (torch.rand(num_samples, dim_s) < pi).float()
        self.s_tilde = self.s + mask * eps

        # Concatenate latents and apply mixing functions
        z1 = torch.cat([self.c, self.s, self.m1], dim=1)
        z2 = torch.cat([self.c, self.s_tilde, self.m2], dim=1)
        self.f1 = MixingFunction(self.dim_total, n_layers=1)
        self.f2 = MixingFunction(self.dim_total, n_layers=1)
        self.x1 = self.f1(z1)
        self.x2 = self.f2(z2)

    def get_dataloader(self, train: bool):
        """Returns a torch.utils.data.DataLoader instance"""
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        c = self.c[idx]
        dataset = TensorDataset(x1, x2, c)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


class Multimodal3DIdent:
    pass
