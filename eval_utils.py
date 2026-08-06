"""Shared helpers: seeding, evaluation, and filter visualization."""
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

def set_seed(seed: int = 42):
    """Freeze all random generators so results repeat exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def report(y_true, y_pred, class_names, title, save_path=None):
    """Print top-1 AND macro-F1; save a readable confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    print(f"{title}: top-1 {acc*100:.2f}%   macro-F1 {mf1*100:.2f}%")
    fig, ax = plt.subplots(figsize=(12, 12))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names,
        normalize="true",         # rows become percentages, not raw counts
        include_values=False,     # hide the numbers: 47x47 is unreadable with them
        xticks_rotation="vertical", colorbar=False, ax=ax)
    ax.set_title(f"{title}  (row-normalized)")
    fig.tight_layout()
    fig.savefig(save_path or f"cm_{title.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)
    return acc, mf1


def plot_conv1_filters(conv_weight, save_path="cnn_conv1_filters.png"):
    """Save a picture of the filters the CNN invented for itself."""
    import torchvision.utils as vutils
    w = conv_weight.detach().cpu().clone()
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)   # rescale to 0..1 so it displays
    grid = vutils.make_grid(w, nrow=8, padding=1)
    plt.figure(figsize=(8, 4)); plt.axis("off")
    plt.title("CNN first-layer filters")
    plt.imshow(grid.permute(1, 2, 0))
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def plot_gabor_bank(bank, save_path="gabor_bank.png"):
    """Save a picture of the filters YOU designed by hand."""
    n = len(bank); cols = 6; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.4, rows*1.4))
    axes = np.ravel(axes)
    for ax, k in zip(axes, bank):
        r = np.real(k); peak = np.abs(r).max() + 1e-9
        ax.imshow(r, cmap="RdBu_r", vmin=-peak, vmax=peak)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Gabor filter bank (real parts)")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)