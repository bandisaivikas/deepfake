# visualize.py - Plots from the paper (Fig. 4 & Fig. 5)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless / no display required
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

SAVE_DIR = "artifacts/plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Training and Validation loss / accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_name: str = "training_curves.png"):
    """
    Reproduce Figure 4: left = loss curves, right = accuracy curves.
    history keys: train_loss, val_loss, train_acc, val_acc
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training and Validation Metrics", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss",    color="royalblue",  linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",      color="darkorange", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Accuracy", color="royalblue",  linewidth=2)
    ax.plot(epochs, history["val_acc"],   label="Val Accuracy",   color="darkorange", linewidth=2)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – t-SNE visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne(embeddings: np.ndarray,
              labels:     np.ndarray,
              title:      str = "t-SNE of Audio Embeddings",
              save_name:  str = "tsne.png",
              perplexity: int = 30,
              n_iter:     int = 1000):
    """
    Reproduce Figure 5: 2-D t-SNE scatter of embeddings coloured by class.
    labels: 0 = real, 1 = fake
    """
    print("[t-SNE] Running dimensionality reduction …")
    tsne   = TSNE(n_components=2, perplexity=perplexity,
                  n_iter=n_iter, random_state=42)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {0: "#2196F3", 1: "#F44336"}
    names  = {0: "Real",    1: "Fake"}

    for cls in [0, 1]:
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[cls], label=names[cls],
                   alpha=0.6, s=20, edgecolors="none")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray,
                          save_name: str = "confusion_matrix.png"):
    cm   = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ROC + EER curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_eer(labels: np.ndarray, probs: np.ndarray,
                 eer: float, eer_threshold: float,
                 save_name: str = "roc_eer.png"):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fnr = 1.0 - tpr

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, color="royalblue", linewidth=2,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # EER (DET-style)
    ax = axes[1]
    ax.plot(fpr, fnr, color="darkorange", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="FAR = FRR line")
    ax.scatter([eer], [eer], c="red", zorder=5,
               label=f"EER = {eer:.4f}\n(threshold={eer_threshold:.4f})")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("False Rejection Rate (FRR)")
    ax.set_title("EER Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {path}")
    return path
