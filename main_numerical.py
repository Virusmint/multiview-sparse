import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.encoders import MLPEncoder, MultiViewEncoders
from src.loss import LpAlignEntropyLoss, SymInfoNCELoss
from src.numerical_experiment.dataset import NumericalDataset
from src.numerical_experiment.latent_space import GaussianSubspace, ProductLatentSpace
from src.numerical_experiment.mixer import MultiViewMixer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        x_views, _ = next(data_iter)
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

    with torch.no_grad():
        for i, (x_views, z_true) in enumerate(data_loader):
            if i >= steps:
                break
            x_views = [x.to(device) for x in x_views]
            z_hats = encoder(x_views)
            z_hat_avg = torch.stack(z_hats, dim=0).mean(dim=0)
            all_z_hat.append(z_hat_avg.cpu().numpy())
            all_z_true.append(z_true.cpu().numpy())

    z_hat = np.concatenate(all_z_hat, axis=0)
    z_true = np.concatenate(all_z_true, axis=0)
    z_hat = StandardScaler().fit_transform(z_hat)

    results = {}
    num_true_latents = z_true.shape[1]
    for i in range(num_true_latents):
        target = z_true[:, i]
        X_train, X_test, y_train, y_test = train_test_split(
            z_hat, target, test_size=0.2, random_state=42
        )
        reg = LinearRegression().fit(X_train, y_train)
        score = r2_score(y_test, reg.predict(X_test))
        results[f'latent_z{i}_R2'] = max(0.0, score)

    results['mean_R2'] = float(np.mean(list(results.values())))
    return results


def build_model(view_configs: List[List[int]], learned_dim: int = 2) -> MultiViewEncoders:
    view_encoders = [
        MLPEncoder(
            input_dim=len(S_k),
            hidden_dims=[128, 128, 128, 128, 128],
            output_dim=learned_dim,
        )
        for S_k in view_configs
    ]
    return MultiViewEncoders(view_encoders)


def negative_l2_similarity(x: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    return -torch.norm(x - y, dim=dim)


def build_criterion(loss_name: str) -> torch.nn.Module:
    loss_name = loss_name.lower()
    if loss_name == 'sym_infonce':
        return SymInfoNCELoss(temperature=1.0, sim_metric=negative_l2_similarity)
    if loss_name == 'lp_align_entropy':
        return LpAlignEntropyLoss(
            p=2,
            tau=1.0,
            use_pow=False,
            align_weight=1.0,
            entropy_weight=1.0,
        )
    raise ValueError(f'Unknown loss_name: {loss_name}')


def run_experiment(
    loss_name: str,
    loader: DataLoader,
    view_configs: List[List[int]],
    device: torch.device,
    output_dir: Path,
    num_epochs: int = 200,
    eval_every: int = 20,
    seed: int = 42,
) -> Tuple[MultiViewEncoders, List[Dict[str, float]]]:
    set_seed(seed)
    model = build_model(view_configs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = build_criterion(loss_name)

    history: List[Dict[str, float]] = []
    print(f"\nStarting experiment: {loss_name} on {device}...")
    for epoch in tqdm(range(1, num_epochs + 1), desc=f'Training [{loss_name}]'):
        loss = train_epoch(model, loader, optimizer, criterion, device)

        row = {'epoch': epoch, 'train_loss': float(loss), 'loss_name': loss_name}
        if epoch % eval_every == 0 or epoch == 1:
            metrics = evaluate(model, loader, device)
            row.update(metrics)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Mean R^2: {metrics['mean_R2']:.4f} | Details: {metrics}")
        history.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f'{loss_name}_numerical_model.pth'
    history_path = output_dir / f'{loss_name}_history.csv'

    torch.save(model.state_dict(), model_path)

    fieldnames = sorted({k for row in history for k in row.keys()})
    with history_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    return model, history


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path('outputs')

    # 1. SETUP WORLD: 6 dimensions
    num_latents = 6
    cov = torch.eye(num_latents)
    latent_world = ProductLatentSpace([GaussianSubspace(dim=num_latents, covariance=cov)])

    # 2. SETUP PHYSICS: 4 views with specific factor overlaps
    view_configs = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 4, 5],
        [0, 1, 2, 3, 5],
        [0, 1, 3, 4, 5],
    ]
    mixer = MultiViewMixer(view_configs)

    # 3. SETUP DATA
    dataset = NumericalDataset(latent_world, mixer, batch_size=4096)
    loader = DataLoader(dataset, batch_size=None)

    # Run both the existing baseline and the fixed theorem-motivated loss.
    experiment_losses = ['sym_infonce', 'lp_align_entropy']
    for loss_name in experiment_losses:
        run_experiment(
            loss_name=loss_name,
            loader=loader,
            view_configs=view_configs,
            device=device,
            output_dir=output_dir,
            num_epochs=200,
            eval_every=20,
            seed=42,
        )
