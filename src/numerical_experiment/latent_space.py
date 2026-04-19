import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import List
from abc import ABC, abstractmethod


# -- Abstract Latent Space Definition --
class LatentSubspace(ABC):
    """
    A single causal factor or a block of factors which may or may not be causally related to each other.
    Each subspace can be arbitrarily complex and have its own distribution and sampling method.
    """

    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def sample(self, batch_size: int, device="cpu"):
        """
        Sample from the subspace according to a distribution.
        """
        raise NotImplementedError("Subclasses must implement the sample method.")


class ProductLatentSpace:
    """
    The global latent space Z formed by concatenating multiple Latent Subspaces.
    Each latent subspace is a block of factors that is independent from the other subspaces,
    but the factors within a subspace may be causally related.
    """

    def __init__(self, subspaces: List[LatentSubspace]):
        self.subspaces = subspaces
        self.dim = sum(s.dim for s in subspaces)

    def sample(self, batch_size: int, device="cpu"):
        # Sample from each subspace and concatenate
        samples = [s.sample(batch_size, device) for s in self.subspaces]
        return torch.cat(samples, dim=-1)


# -- Concrete Latent Subspaces --
class GaussianSubspace(LatentSubspace):
    def __init__(self, dim: int, mean=0.0, covariance=None):
        super().__init__(dim)
        self.mean = mean
        self.covariance = (
            covariance if covariance is not None else torch.eye(dim)
        )  # Default to identity covariance

    def sample(self, batch_size: int, device="cpu"):
        dist = MultivariateNormal(
            torch.full((self.dim,), self.mean).to(device),
            self.covariance.to(device),
        )
        return dist.sample((batch_size,))


class UniformSubspace(LatentSubspace):
    def __init__(self, dim: int, low=0.0, high=1.0):
        super().__init__(dim)
        self.low = low
        self.high = high

    def sample(self, batch_size: int, device="cpu"):
        return (
            torch.rand((batch_size, self.dim), device=device) * (self.high - self.low)
            + self.low
        )
