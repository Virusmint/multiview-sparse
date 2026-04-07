import torch
from torch.utils.data import IterableDataset
from src.numerical_experiment.latent_space import ProductLatentSpace
from src.numerical_experiment.mixer import MultiViewMixer

from typing import List, Tuple, Iterator


class NumericalDataset(IterableDataset):
    def __init__(
        self,
        latent_space: ProductLatentSpace,
        multi_mixer: MultiViewMixer,
        batch_size: int = 4096,
    ):
        self.latent_space = latent_space
        self.multi_mixer = multi_mixer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], torch.Tensor]]:
        """Infinite generator of batches of multi-view data and their corresponding latent factors."""
        while True:
            z_global = self.latent_space.sample(self.batch_size)
            views = self.multi_mixer(z_global)
            yield views, z_global
