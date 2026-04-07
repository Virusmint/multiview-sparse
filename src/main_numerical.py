import torch
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from dataset import NumericalDataset
from models import BimodalEncoders
from loss import SymmetricedInfoNCE


def train_model(
    model, train_dataloader, val_dataloader, optimizer, criterion, epochs, device
):
    model.to(device)

    try:
        for epoch in range(epochs):
            # Training step
            model.train()
            train_loss = 0.0
            for x1, x2, _ in train_dataloader:
                x1, x2 = x1.to(device), x2.to(device)
                # Forward pass
                z1_hat, z2_hat = model(x1, x2)
                # Compute loss
                loss = criterion(z1_hat, z2_hat)
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_dataloader)

            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x1, x2, _ in val_dataloader:
                    x1, x2 = x1.to(device), x2.to(device)
                    z1_hat, z2_hat = model(x1, x2)
                    val_loss += criterion(z1_hat, z2_hat).item()
            avg_val_loss = val_loss / len(val_dataloader)

            # Print performance
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Exiting gracefully.")


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

    model_path = "./models/bimodal_encoder_l2.pth"
    model_output_path = "./models/test.pth"

    # Setup Data
    dim_c, dim_s, dim_m = 4, 4, 2
    dim_total = dim_c + dim_s + dim_m
    data = NumericalDataset(
        num_samples=200000,
        dim_c=dim_c,
        dim_s=dim_s,
        dim_m=dim_m,
        causal=True,
        pi=1.0,
        batch_size=8192,
    )

    # Setup model
    encoder_layers = [dim_total, 128, 512, 512, 512, 512, 128, dim_c]  # 7-layer mlp
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

    # model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    train_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        epochs=500,
        device=device,
    )

    # Evaluate performance of trainer encoders
    print("Evaluating Block-Identifiability...")
    evaluate_block_identifiability(model, data.val_dataloader(), device=device)

    # Save model
    torch.save(model.state_dict(), model_output_path)
    print(f"Saved model to {model_output_path}")
