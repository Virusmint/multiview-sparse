import torch
import torch.nn as nn
from typing import List


class HardConcreteGate(nn.Module):
    def __init__(
        self,
        dim: int,
        init_mean: float = 0.5,
        init_std: float = 0.01,
        beta: float = 0.66,
        gamma: float = -0.1,
        zeta: float = 1.1,
        decay_rate: float = 0.96,
        min_beta: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.decay_rate = decay_rate
        self.min_beta = min_beta

        # We parameterize log_alpha.
        # Initializing it so that the initial gate values are slightly open.
        self.log_alpha = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.log_alpha, mean=init_mean, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Stochastic gate values using the hard concrete distribution during training
            u = torch.rand_like(self.log_alpha)  # Uniform(0, 1)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta
            )  # Gumbel-sigmoid reparameterization
            s_bar = (
                s * (self.zeta - self.gamma) + self.gamma
            )  # Stretching to [gamma, zeta]
            z = torch.clamp(s_bar, min=0, max=1)  # Hard concrete gate
        else:
            # Deterministic gate values during evaluation
            s = torch.sigmoid(self.log_alpha)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            z = torch.clamp(s_bar, min=0, max=1)
        return x * z  # Apply the gate to the input

    def anneal_temperature(self):
        """
        Anneals the beta parameter to encourage harder gating over time.
        """
        self.beta = max(self.beta * self.decay_rate, self.min_beta)

    def get_l0_penalty(self) -> torch.Tensor:
        """
        Computes the expected L0 norm of the gates for sparsity regularization.
        P(z > 0) = Sigmoid(log_alpha - beta * log(-gamma / zeta))
        """
        term = self.log_alpha - self.beta * torch.log(
            torch.tensor(-self.gamma / self.zeta, device=self.log_alpha.device)
        )
        p_active = torch.sigmoid(term)
        return p_active.sum()  # Sum over all gates for total expected L0 norm

    @torch.no_grad()
    def get_values(self) -> torch.Tensor:
        """
        Computes the deterministic gate values (z) for evaluation.
        z = clamp(sigmoid(log_alpha) * (zeta - gamma) + gamma, 0, 1)
        """
        s = torch.sigmoid(self.log_alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, min=0, max=1)
        return z
