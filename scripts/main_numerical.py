import sys
import os
import argparse
import json
import torch
from torch.distributions import Wishart
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Numerical M3DI: Causal Factor Independence Experiment"
    )

    # Latent Structure Arguments
    parser.add_argument(
        "--latent-mode",
        type=str,
        choices=["indep", "dep"],
        default="indep",
        help="Choose 'indep' for identity covariance or 'dep' for Wishart-sampled correlations.",
    )
    parser.add_argument(
        "--wishart-df",
        type=int,
        default=None,
        help="Degrees of freedom for Wishart distribution. Defaults to num_latents + 1.",
    )
    parser.add_argument(
        "--view-configs",
        type=json.loads,
        default="[[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 1, 8], [0, 1, 9]]",
        help="JSON string representing the list of lists for view configurations.",
    )

    # Training Hyperparameters
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-sparse", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--estimated-dim", type=int, default=6)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])

    # Paths & Setup
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--checkpoint-path", type=str, default="checkpoint/numerical_model.pth"
    )
    parser.add_argument(
        "--gate-plot-path", type=str, default="figures/gate_history.png"
    )

    return parser.parse_args()


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
            # Average the representations from each view as proxy for the shared content.
            # Alternatively, we could eval each view seperately and/or pool them differently, but this is simpler.
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
        X_train, X_test, y_train, y_test = train_test_split(
            z_hat, target, test_size=0.2, random_state=42
        )
        # Fit Linear Probe
        reg = LinearRegression().fit(X_train, y_train)
        score = r2_score(y_test, reg.predict(X_test))

        results[f"latent_z{i}_R2"] = max(0, score)  # Clamp at 0 for readability

    return results


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # 1. SETUP PHYSICS & LATENT SPACE
    view_configs = args.view_configs
    num_latents = len(set().union(*view_configs))

    if args.latent_mode == "indep":
        cov = torch.eye(num_latents)
    else:
        df = args.wishart_df if args.wishart_df else num_latents + 1
        wishart_dist = Wishart(df=df, covariance_matrix=torch.eye(num_latents))
        cov = wishart_dist.sample()

    latent_space = ProductLatentSpace(
        [GaussianSubspace(dim=num_latents, covariance=cov)]
    )
    mixer = MultiViewMixer(view_configs)

    # 2. SETUP DATA
    dataset = NumericalDataset(latent_space, mixer, batch_size=args.batch_size)
    data_loader = DataLoader(dataset, batch_size=None)

    # 3. SETUP MODEL
    view_encoders = [
        MLPEncoder(
            input_dim=len(S_k),
            hidden_dims=args.hidden_dims,
            output_dim=args.estimated_dim,
        )
        for S_k in view_configs
    ]
    model = MultiViewEncoders(view_encoders).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. SETUP LOSS
    criterion = SparseInfoNCELoss(
        encoders=model,
        lambda_=args.lambda_sparse,
        temperature=args.temperature,
        sim_metric=cosine_sim,
    )

    # 5. TRAINING LOOP
    gate_history = np.zeros((args.num_epochs, args.estimated_dim))
    print(f"Starting experiment on {device}...")
    pbar = tqdm(range(1, args.num_epochs + 1), desc="Training")
    epoch = 0
    try:
        for epoch in pbar:
            criterion.set_sparsity(
                epoch <= args.warmup_epochs
            )  # Disable sparsity penalty during warmup

            loss, gate_values = train_epoch(
                model, data_loader, optimizer, criterion, device, steps=args.train_steps
            )

            pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "gate values": f"{np.round(gate_values, 3)}",
                }
            )
            gate_history[epoch - 1] = gate_values
            if epoch % 20 == 0 or epoch == 1:
                r2 = evaluate(model, data_loader, device, steps=args.eval_steps)
                print(f"\nEpoch {epoch:03d} |  Block R^2: {r2}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        # Save model checkpoint
        checkpoint_path = Path(args.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

        # Save gate history plot
        gate_plot_path = Path(args.gate_plot_path)
        gate_plot_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure we only plot the epochs that actually ran if interrupted
        plot_length = epoch if epoch > 0 else 1
        plot_gate_history(
            gate_history[:plot_length],
            save_path=str(gate_plot_path),
        )
        print(f"Gate history plot saved to {gate_plot_path}")


if __name__ == "__main__":
    main()
