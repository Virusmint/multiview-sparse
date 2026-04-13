import matplotlib.pyplot as plt
import numpy as np
import os


def plot_gate_history(
    gate_history: np.ndarray,
    save_path: str = "plots/gate_history.png",
    use_heatmap: bool = False,
    show: bool = False,
):
    """
    Plots the evolution of sparsity gates.

    Args:
        gate_history: Numpy array of shape (epochs, num_dimensions)
        save_path: Where to save the image
        use_heatmap: If True, renders a heatmap. If False, renders a line plot.
        show: Whether to pop up the window
    """
    epochs, dims = gate_history.shape
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))

    if use_heatmap:
        # Heatmap: X=Epochs, Y=Dimensions
        # aspect='auto' prevents the heatmap from being a tiny thin strip
        im = plt.imshow(
            gate_history.T,
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            extent=[0, gate_history.shape[0] - 1, 1, gate_history.shape[1]],
        )
        plt.colorbar(im, label="Gate Value (Sparsity)")
        plt.ylabel("Latent Dimension Index")
        plt.xlabel("Epoch")
    else:
        # Line Plot: Each line is one dimension
        for i in range(dims):
            plt.plot(gate_history[:, i], label=f"Dim {i + 1}", linewidth=1.5)

        plt.ylabel("Gate Values")
        plt.xlabel("Epoch")
        plt.ylim(-0.05, 1.05)  # Keep scale consistent
        plt.legend(loc="best")
        plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()
    print(f"Plot saved to {save_path}")
