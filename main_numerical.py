import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np

from tqdm import tqdm
from typing import Dict

# Add src to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.numerical_experiment.dataset import NumericalDataset
from src.numerical_experiment.latent_space import ProductLatentSpace, GaussianSubspace
from src.numerical_experiment.mixer import MultiViewMixer
from src.loss import LpAlignEntropyLoss, SymInfoNCELoss
from src.encoders import MultiViewEncoders, MLPEncoder


def train_epoch(
    encoder: MultiViewEncoders,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    steps: int = 100,
) -> float:
    encoder.train()
    epoch_loss = 0.0
    data_iter = iter(data_loader)
    for _ in range(steps):
        x_views, z_true = next(data_iter)
        x_views = [x.to(device) for x in x_views]

        optimizer.zero_grad()
        z_hats = encoder(x_views)
        loss = criterion(z_hats)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / steps


def evaluate(
    encoder: MultiViewEncoders,
    data_loader: DataLoader,
    device: torch.device,
    steps: int = 20,
) -> Dict[str, float]:

    encoder.eval()
    all_z_hat, all_z_true = [], []
    # 1. Collect Data
    with torch.no_grad():
        for i, (x_views, z_true) in enumerate(data_loader):
            if i >= steps:
                break
            x_views = [x.to(device) for x in x_views]
            # Get representations from all views
            z_hats = encoder(x_views)
            # Paper style: Use the content from one view or average them
            # Let's average them as a proxy for the 'shared' content
            z_hat_avg = torch.stack(z_hats, dim=0).mean(dim=0)
            all_z_hat.append(z_hat_avg.cpu().numpy())
            all_z_true.append(z_true.cpu().numpy())

    # Concatenate into large matrices
    z_hat = np.concatenate(all_z_hat, axis=0)  # [N, latent_dim_learned]
    z_true = np.concatenate(all_z_true, axis=0)  # [N, latent_dim_ground_truth]
    # 2. Standardize
    z_hat = StandardScaler().fit_transform(z_hat)

    # 3. Dimension-wise Evaluation
    results = {}
    num_true_latents = z_true.shape[1]

    for i in range(num_true_latents):
        target = z_true[:, i]
        # Split to validate generalization
        X_train, X_test, y_train, y_test = train_test_split(
            z_hat, target, test_size=0.2, random_state=42
        )
        # Fit Linear Probe
        reg = LinearRegression().fit(X_train, y_train)
        score = r2_score(y_test, reg.predict(X_test))

        results[f"latent_z{i}_R2"] = max(0, score)  # Clamp at 0 for readability

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. SETUP WORLD: 6 dimensions (3 independent, 3 in a causal chain)
    num_latents = 6
    cov = torch.eye(num_latents)
    latent_world = ProductLatentSpace(
        [GaussianSubspace(dim=num_latents, covariance=cov)]
    )

    # 2. SETUP PHYSICS: 4 Views with specific factor overlaps
    view_configs = [[0, 1, 2, 3, 4], [0, 1, 2, 4, 5], [0, 1, 2, 3, 5], [0, 1, 3, 4, 5]]
    mixer = MultiViewMixer(view_configs)

    # 3. SETUP DATA
    dataset = NumericalDataset(latent_world, mixer, batch_size=4096)
    loader = DataLoader(dataset, batch_size=None)  # batch_size=None for IterableDataset

    # 4. SETUP MODEL & OPTIMIZER
    view_encoders = [
        MLPEncoder(
            input_dim=len(S_k),
            hidden_dims=[128, 128, 128, 128, 128],
            output_dim=2,
        )
        for S_k in view_configs
    ]
    model = MultiViewEncoders(view_encoders).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # criterion = LpAlignEntropyLoss(p=2, tau=1.0)
    def negative_l2_similarity(x, y, dim):
        return -torch.norm(x - y, dim=dim)

    criterion = SymInfoNCELoss(temperature=1.0, sim_metric=negative_l2_similarity)

    # 5. TRAINING LOOP
    print(f"Starting experiment on {device}...")
    for epoch in tqdm(range(1, 201), desc="Training"):
        loss = train_epoch(model, loader, optimizer, criterion, device)

        if epoch % 20 == 0 or epoch == 1:
            r2 = evaluate(model, loader, device)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Block R^2: {r2}")

    torch.save(model.state_dict(), "scripts/numerical_model.pth")
