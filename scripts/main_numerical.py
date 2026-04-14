import sys
import os
import torch
from torch.distributions import Wishart
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from typing import Dict, List, Tuple

# Add src to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.numerical_experiment.dataset import NumericalDataset
from src.numerical_experiment.latent_space import ProductLatentSpace, GaussianSubspace
from src.numerical_experiment.mixer import MultiViewMixer
from src.loss import SparseInfoNCELoss
from src.encoders import MultiViewEncoders, MLPEncoder
from src.metrics import cosine_sim
from src.utils.plotting import plot_gate_history


def train_epoch(
    encoders: MultiViewEncoders,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    steps: int = 100,  # Number of gradient steps per epoch
) -> Tuple[float, List[int]]:
    encoders.train()
    epoch_loss = 0.0
    data_iter = iter(data_loader)
    for _ in range(steps):
        x_views, z_true = next(data_iter)
        x_views = [x.to(device) for x in x_views]

        optimizer.zero_grad()
        z_hats = encoders(x_views)  # Get representations from all views
        loss = criterion(z_hats)

        # Optimize
        loss.backward()
        optimizer.step()

        # Logging
        epoch_loss += loss.item()
    gate_values = (
        encoders.get_gate_values().cpu().numpy().tolist()
    )  # Get current gate values for logging
    return epoch_loss / steps, gate_values


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
            # Use the content from one view or average them
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

    # 1. SETUP PHYSICS: Indices 0 and 1 are content
    view_configs = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 5],
        [0, 1, 6],
        [0, 1, 7],
        [0, 1, 8],
        [0, 1, 9],
    ]
    num_latents = len(set().union(*view_configs))  # Num unique latents
    cov = Wishart(
        df=num_latents + 1, covariance_matrix=torch.eye(num_latents)
    ).sample()  # Random covariance
    # cov = torch.eye(num_latents)  # Independent latents for simplicity
    latent_space = ProductLatentSpace(
        [GaussianSubspace(dim=num_latents, covariance=cov)]
    )
    mixer = MultiViewMixer(view_configs)

    # 3. SETUP DATA
    dataset = NumericalDataset(latent_space, mixer, batch_size=4096)
    data_loader = DataLoader(
        dataset, batch_size=None
    )  # batch_size=None for IterableDataset

    # 4. SETUP MODEL & OPTIMIZER
    estimated_dim = 6  # Overestimate the latent dimension to test if the model prune the irrelevant dimensions
    view_encoders = [
        MLPEncoder(
            input_dim=len(S_k),
            hidden_dims=[128, 128],
            output_dim=estimated_dim,
        )
        for S_k in view_configs
    ]
    model = MultiViewEncoders(view_encoders).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5. SETUP LOSS
    lambda_ = 0.1  # Regularization strength for sparsity
    # criterion = LpAlignEntropyLoss(p=2, tau=1.0)
    # criterion = SymInfoNCELoss(temperature=1.0, sim_metric=cosine_sim)
    criterion = SparseInfoNCELoss(
        encoders=model, lambda_=lambda_, temperature=2.0, sim_metric=cosine_sim
    )

    # 6. TRAINING LOOP
    warmup_epochs = 0  # Number of epochs to train without sparsity penalty
    num_epochs = 60
    gate_history = np.zeros((num_epochs, estimated_dim))

    print(f"Starting experiment on {device}...")
    pbar = tqdm(range(1, num_epochs + 1), desc="Training")
    try:
        for epoch in pbar:
            criterion.set_sparsity(
                epoch < warmup_epochs
            )  # Disable sparsity penalty during warmup

            loss, gate_values = train_epoch(
                model, data_loader, optimizer, criterion, device
            )
            model.anneal_temperature()  # Anneal temperature for hard concrete gates

            pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "gate values": f"{np.round(gate_values, 3)}",
                }
            )
            gate_history[epoch - 1] = gate_values
            if epoch % 20 == 0 or epoch == 1:
                r2 = evaluate(model, data_loader, device)
                print(f"Epoch {epoch:03d} |  Block R^2: {r2}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
    finally:
        torch.save(model.state_dict(), "checkpoint/numerical_model.pth")
        print("Model saved to checkpoint/numerical_model.pth")
        plot_gate_history(
            gate_history,
            save_path="figures/numerical_gate_history.png",
        )
