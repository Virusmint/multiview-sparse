import argparse
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

from typing import Dict, List, Tuple

from tqdm import tqdm

# Add src to path so we can import project modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encoders import ImageEncoderResNet, MultiViewEncoders, TextEncoder2D
from src.loss import SparseInfoNCELoss
from src.utils.plotting import plot_gate_history
from src.metrics import cosine_sim
from src.multimodal_experiment.datasets import Multimodal3DIdent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_views(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> List[torch.Tensor]:
    return [batch["image"].to(device), batch["text"].to(device)]


def train_epoch(
    model: MultiViewEncoders,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, List[float]]:
    model.train()
    epoch_loss = 0.0
    num_batches = len(data_loader)

    for batch in data_loader:
        x_views = to_views(batch, device)
        optimizer.zero_grad()
        z_hats = model(x_views)
        loss = criterion(z_hats)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    gate_values = model.get_gate_values().detach().cpu().numpy().tolist()
    return epoch_loss / num_batches, gate_values


def get_representations_and_labels(
    model: MultiViewEncoders, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Helper to extract z_hat and ground truth labels for a full dataset."""
    model.eval()
    all_z_hat = []

    # Initialize dictionary to hold ground truth labels
    factors_info = dataloader.dataset.FACTORS["image"]
    all_z_true = {name: [] for name in factors_info.values()}

    with torch.no_grad():
        for batch in dataloader:
            x_views = to_views(batch, device)
            z_hats = model(x_views)

            # Average representations across views for the shared content
            z_hat_avg = torch.stack(z_hats, dim=0).mean(dim=0)
            all_z_hat.append(z_hat_avg.cpu().numpy())

            # Collect ground truth labels
            for name in all_z_true.keys():
                all_z_true[name].append(batch["z_image"][name].cpu().numpy())

    # Concatenate all batches
    z_hat = np.concatenate(all_z_hat, axis=0)
    z_true = {name: np.concatenate(vals, axis=0) for name, vals in all_z_true.items()}

    return z_hat, z_true


def evaluate(
    model: MultiViewEncoders,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates identifiability by fitting probes on Val and scoring on Test."""
    print("Extracting representations for validation and test sets...")
    z_val, s_val = get_representations_and_labels(model, val_loader, device)
    z_test, s_test = get_representations_and_labels(model, test_loader, device)

    # 1. Standardize based ONLY on validation statistics
    scaler = StandardScaler().fit(z_val)
    z_val = scaler.transform(z_val)
    z_test = scaler.transform(z_test)

    discrete_names = set(val_loader.dataset.DISCRETE_FACTORS["image"].values())
    shared_names = set(val_loader.dataset.FACTORS["text"].values())
    results = {}

    print("Fitting linear probes...")
    # 2. Fit Probes per Factor
    for name in s_val.keys():
        y_val = s_val[name]
        y_test = s_test[name]

        tag = "SHARED" if name in shared_names else "STYLE "

        if name in discrete_names:
            # CATEGORICAL PROBE (Classification)
            clf = LogisticRegression(max_iter=1000).fit(z_val, y_val)
            acc = accuracy_score(y_test, clf.predict(z_test))
            results[f"[{tag}] {name} (Acc)"] = acc
        else:
            # CONTINUOUS PROBE (Regression)
            reg = LinearRegression().fit(z_val, y_val)
            r2 = r2_score(y_test, reg.predict(z_test))
            results[f"[{tag}] {name} (R2)"] = max(0, r2)  # Clamp negative R2

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal M3DI training with hard-concrete gating."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--estimated-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--lambda-sparse", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--checkpoint-path", type=str, default="checkpoint/multimodal_model.pth"
    )
    parser.add_argument(
        "--gate-plot-path",
        type=str,
        default="figures/multimodal_gate_history.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mean_per_channel = [0.4327, 0.2689, 0.2839]
    std_per_channel = [0.1201, 0.1457, 0.1082]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean_per_channel, std_per_channel),
        ]
    )
    # 1. SETUP DATA
    train_dataset = Multimodal3DIdent(args.data_root, mode="train", transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Setup Val and Test Datasets (Sharing the train vocabulary)
    vocab_path = train_dataset.vocab_filepath
    val_dataset = Multimodal3DIdent(
        args.data_root, mode="val", transform=transform, vocab_filepath=vocab_path
    )
    test_dataset = Multimodal3DIdent(
        args.data_root, mode="test", transform=transform, vocab_filepath=vocab_path
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # 2. SETUP MODEL & OPTIMIZER
    image_encoder = ImageEncoderResNet(
        output_dim=args.estimated_dim,
        hidden_dim=args.hidden_size,
    )
    text_encoder = TextEncoder2D(
        input_dim=train_dataset.vocab_size,
        output_dim=args.estimated_dim,
        sequence_length=train_dataset.max_sequence_length,
    )

    model = MultiViewEncoders(
        view_encoders=[image_encoder, text_encoder],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. SETUP LOSS
    criterion = SparseInfoNCELoss(
        encoders=model,
        lambda_=args.lambda_sparse,
        temperature=args.temperature,
        sim_metric=cosine_sim,
    )

    # 4. TRAINING LOOP
    gate_history = np.zeros((args.num_epochs, args.estimated_dim))
    print(f"Starting multimodal experiment on {device}...")
    pbar = tqdm(range(1, args.num_epochs + 1), desc="Training")
    epoch = 0
    try:
        for epoch in pbar:
            criterion.set_sparsity(epoch <= args.warmup_epochs)

            loss, gate_values = train_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "gate values": f"{np.round(gate_values, 3)}",
                }
            )
            gate_history[epoch - 1] = gate_values
            if epoch % 25 == 0:
                results = evaluate(model, val_loader, test_loader, device)
                result_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items()])
                print(f"Epoch {epoch:03d} |  {result_str}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
    finally:
        # Save model checkpoint
        checkpoint_path = Path(args.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to `{checkpoint_path}`.")

        # Save gate history plot
        gate_plot_path = Path(args.gate_plot_path)
        gate_plot_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure we only plot the epochs that actually ran if interrupted
        plot_length = epoch if epoch > 0 else 1
        plot_gate_history(
            gate_history[:plot_length],
            save_path=str(gate_plot_path),
        )
        print(f"Saved gate history plot to `{gate_plot_path}`.")


if __name__ == "__main__":
    import nltk

    nltk.download("punkt_tab", quiet=True)
    main()
