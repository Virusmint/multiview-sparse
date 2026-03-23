import torch
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from dataset import NumericalDataset
from models import BimodalEncoders
from loss import SymmetricedInfoNCE
from trainer import Trainer


def train_model(model, dataloader, optimizer, criterion, epochs, device):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (x1, x2, _) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            # Forward pass
            z1_hat, z2_hat = model(x1, x2)
            # Compute loss
            loss = criterion(z1_hat, z2_hat)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f}")


def evaluate_block_identifiability(model, dataloader, device):
    model.eval()
    model.to(device)

    c_hat_list, c_true_list = [], []

    with torch.no_grad():
        for x1, x2, c_true in dataloader:
            x1 = x1.to(device)
            # Use Encoder 1 to extract the representation
            c_hat = model.enc1(x1)

            c_hat_list.append(c_hat.cpu().numpy())
            c_true_list.append(c_true.numpy())

    c_hat = np.concatenate(c_hat_list, axis=0)
    c_true = np.concatenate(c_true_list, axis=0)

    # Linear regression to test block-identifiability
    linear_reg = LinearRegression()
    linear_reg.fit(c_hat, c_true)

    c_pred = linear_reg.predict(c_hat)
    r2 = r2_score(c_true, c_pred)
    print(f"\nFinal Linear Regression R^2 Score: {r2:.4f}")
    return r2


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup Data
    dim_c, dim_s, dim_m = 4, 4, 2
    dim_total = dim_c + dim_s + dim_m
    data = NumericalDataset(
        num_samples=200000,
        dim_c=dim_c,
        dim_s=dim_s,
        dim_m=dim_m,
        causal=True,
        pi=0.5,
        batch_size=8192,
    )

    # Setup model
    encoder_layers = [dim_total, 128, 128, 128, 128, 128, dim_c]  # 7-layer mlp
    criterion = SymmetricedInfoNCE(temperature=1.0)
    model = BimodalEncoders(
        layers1=encoder_layers, layers2=encoder_layers, criterion=criterion
    )

    # # Setup Trainer
    # criterion = SymmetricedInfoNCE()
    # trainer = Trainer(max_epochs=100, device=device)
    #
    # # Fit model
    # print("Starting Training...")
    # trainer.fit(model, data)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = data.train_dataloader()
    test_loader = data.val_dataloader()
    train_model(model, train_loader, optimizer, criterion, epochs=1000, device=device)

    # Evaluate performance of trainer encoders
    print("Evaluating Block-Identifiability...")
    evaluate_block_identifiability(model, data.val_dataloader(), device=device)
