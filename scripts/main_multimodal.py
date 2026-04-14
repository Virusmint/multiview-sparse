import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add src to path so we can import project modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encoders import ImageEncoderResNet, MultiViewEncoders, TextEncoder2D
from src.loss import SparseInfoNCELoss, SymInfoNCELoss
from src.utils.plotting import plot_gate_history
from src.utils.sim_metric import cosine_sim


def load_external_multimodal_dataset_class():
    external_path = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "multimodal-repo"
        / "datasets.py"
    )
    spec = importlib.util.spec_from_file_location("external_multimodal_datasets", external_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load dataset module from `{external_path}`.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Multimodal3DIdent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_views(batch: Dict[str, torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [batch["image"].to(device), batch["text"].to(device)]


def train_epoch(
    model: MultiViewEncoders,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    steps: int,
) -> Tuple[float, List[float]]:
    model.train()
    epoch_loss = 0.0
    data_iter = iter(data_loader)

    for _ in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        x_views = to_views(batch, device)
        optimizer.zero_grad()
        z_hats = model(x_views)
        loss = criterion(z_hats)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    gate_values = model.get_gate_values().detach().cpu().numpy().tolist()
    return epoch_loss / steps, gate_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal M3DI training with hard-concrete gating.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--estimated-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--lambda-sparse", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sparsity", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoint/multimodal_model.pth")
    parser.add_argument(
        "--gate-plot-path",
        type=str,
        default="figures/multimodal_gate_history_heatmap.png",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    Multimodal3DIdent = load_external_multimodal_dataset_class()

    mean_per_channel = [0.4327, 0.2689, 0.2839]
    std_per_channel = [0.1201, 0.1457, 0.1082]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean_per_channel, std_per_channel),
        ]
    )
    train_dataset = Multimodal3DIdent(args.data_root, mode="train", transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    image_encoder = ImageEncoderResNet(
        output_dim=args.estimated_dim,
        hidden_size=args.hidden_size,
    )
    text_encoder = TextEncoder2D(
        input_size=train_dataset.vocab_size,
        output_size=args.estimated_dim,
        sequence_length=train_dataset.max_sequence_length,
    )

    use_sparsity = not args.no_sparsity
    model = MultiViewEncoders(
        view_encoders=[image_encoder, text_encoder],
        use_sparsity=use_sparsity,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if use_sparsity:
        criterion: torch.nn.Module = SparseInfoNCELoss(
            encoders=model,
            lambda_=args.lambda_sparse,
            temperature=args.temperature,
            sim_metric=cosine_sim,
        )
    else:
        criterion = SymInfoNCELoss(
            temperature=args.temperature,
            sim_metric=cosine_sim,
        )

    gate_history = np.zeros((args.num_epochs, args.estimated_dim))
    print(f"Starting multimodal experiment on {device}...")
    print(
        {
            "estimated_dim": args.estimated_dim,
            "use_sparsity": use_sparsity,
            "vocab_size": train_dataset.vocab_size,
            "max_sequence_length": train_dataset.max_sequence_length,
        }
    )

    for epoch in tqdm(range(1, args.num_epochs + 1), desc="Training"):
        if isinstance(criterion, SparseInfoNCELoss):
            criterion.set_sparsity(warmup=epoch <= args.warmup_epochs)

        loss, gate_values = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            steps=args.steps_per_epoch,
        )
        gate_history[epoch - 1] = gate_values
        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Gate {np.round(gate_values, 3)}")

        if args.dry_run:
            break

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to `{checkpoint_path}`.")

    gate_plot_path = Path(args.gate_plot_path)
    gate_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_gate_history(
        gate_history[:epoch],
        save_path=str(gate_plot_path),
        use_heatmap=True,
    )
    print(f"Saved gate history plot to `{gate_plot_path}`.")


if __name__ == "__main__":
    main()
