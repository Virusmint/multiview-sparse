import torch
from src.numerical_experiment.mixer import MultiViewMixer
from src.numerical_experiment.dataset import NumericalDataset
from src.numerical_experiment.latent_space import ProductLatentSpace, GaussianSubspace

# 1. Setup the Latent World
# Example: 3 independent factors, 3 causally linked factors
world = ProductLatentSpace(
    [
        GaussianSubspace(dim=2, mean=0.0, cov=torch.eye(2)),  # Independent block 1
        GaussianSubspace(dim=1, mean=0.0, cov=torch.eye(1)),  # Independent block 2
        GaussianSubspace(dim=3, mean=0.0, cov=torch.eye(3)),  # Causally linked block
    ]
)

# 2. Setup the "Physics" (The mixing functions)
# Each view sees specific latent indices
view_configs = [[0, 1, 3], [1, 2, 4], [3, 4, 5]]
physics_engine = MultiViewMixer(view_configs)

# 3. Create the Dataset
dataset = NumericalDataset(world, physics_engine, batch_size=4096)
dataloader = torch.utils.data.DataLoader(dataset)

# 4. Test a single batch
x_views, z_true = next(iter(dataloader))

print(f"Sampled reality: {z_true.shape}")  # [512, 6]
for i, obs in enumerate(x_views):
    print(f"View {i} observation: {obs.shape}")  # [512, 16]
