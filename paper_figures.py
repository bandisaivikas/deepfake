"""
paper_figures.py
================
Generate ALL figures needed for the deepfake audio detection paper.
Run this script (or import individual functions) after training is complete.

Usage (notebook cell):
    import importlib, paper_figures
    importlib.reload(paper_figures)
    paper_figures.generate_all(
        model=model,
        history=history,
        dataset=dataset,
        results=results,
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
    )
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from sklearn.decomposition import PCA
import librosa
import librosa.display

import config
from dataset import load_audio, wav_to_log_mel, DeepfakeAudioDataset
from models import GRUEmbedder, FineTunedModel

# ── Output directory ──────────────────────────────────────────────────────────
SAVE_DIR = "artifacts/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

def _save(fig, name, dpi=200):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Dataset Class Distribution (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def fig_dataset_distribution(dataset: DeepfakeAudioDataset):
    """Bar chart showing real vs. fake sample counts."""
    labels = [s[1] for s in dataset.samples]
    n_real = labels.count(0)
    n_fake = labels.count(1)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Real", "Fake"], [n_real, n_fake],
                  color=["#2196F3", "#F44336"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, [n_real, n_fake]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Dataset Class Distribution", fontsize=13)
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, max(n_real, n_fake) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, "fig1_dataset_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Sample Waveforms (Real vs Fake, side by side)
# ─────────────────────────────────────────────────────────────────────────────

def fig_sample_waveforms(dataset: DeepfakeAudioDataset):
    """Plot one real and one fake waveform side by side."""
    real_path = next(p for p, l in dataset.samples if l == 0)
    fake_path = next(p for p, l in dataset.samples if l == 1)

    real_wav = load_audio(real_path)
    fake_wav = load_audio(fake_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    t = np.linspace(0, config.DURATION, len(real_wav))

    for ax, wav, title, color in zip(
        axes,
        [real_wav, fake_wav],
        ["Real Audio", "Fake (Deepfake) Audio"],
        ["#2196F3", "#F44336"]
    ):
        ax.plot(t, wav, color=color, linewidth=0.5, alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Sample Audio Waveforms", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "fig2_sample_waveforms.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Log-Mel Spectrograms (Real vs Fake)
# ─────────────────────────────────────────────────────────────────────────────

def fig_mel_spectrograms(dataset: DeepfakeAudioDataset):
    """Show log-Mel spectrograms for one real and one fake sample."""
    real_path = next(p for p, l in dataset.samples if l == 0)
    fake_path = next(p for p, l in dataset.samples if l == 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, path, title in zip(
        axes,
        [real_path, fake_path],
        ["Real Audio – Log-Mel Spectrogram", "Fake Audio – Log-Mel Spectrogram"]
    ):
        wav = load_audio(path)
        mel = librosa.feature.melspectrogram(
            y=wav, sr=config.SAMPLE_RATE,
            n_mels=config.N_MELS, n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            fmin=config.F_MIN, fmax=config.F_MAX
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(
            log_mel, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
            x_axis="time", y_axis="mel", fmin=config.F_MIN, fmax=config.F_MAX,
            ax=ax, cmap="magma"
        )
        ax.set_title(title, fontsize=11)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.suptitle("Log-Mel Spectrograms: Real vs Fake", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "fig3_mel_spectrograms.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Training & Validation Curves (Loss + Accuracy)
# ─────────────────────────────────────────────────────────────────────────────

def fig_training_curves(history: dict):
    """Reproduce paper Figure 4: loss and accuracy curves."""
    if not history.get("train_loss"):
        print("  [SKIP] No training history available.")
        return None

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training and Validation Metrics", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#1565C0", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#E65100", linewidth=2, linestyle="--")
    ax.set_title("Cross-Entropy Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, [a*100 for a in history["train_acc"]], label="Train Acc", color="#1565C0", linewidth=2)
    ax.plot(epochs, [a*100 for a in history["val_acc"]],   label="Val Acc",   color="#E65100", linewidth=2, linestyle="--")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    return _save(fig, "fig4_training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def fig_confusion_matrix(y_true, y_pred):
    """Normalised and raw confusion matrices side by side."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Confusion Matrix", fontsize=13, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".2%"],
        ["Raw Counts", "Normalised (Row %)"]
    ):
        disp = ConfusionMatrixDisplay(data, display_labels=["Real", "Fake"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=fmt)
        ax.set_title(title)

    plt.tight_layout()
    return _save(fig, "fig5_confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — ROC Curve + AUC
# ─────────────────────────────────────────────────────────────────────────────

def fig_roc_curve(y_true, y_prob, eer=None, eer_threshold=None):
    """ROC curve with AUC annotation and optional EER point."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fnr = 1.0 - tpr

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ROC and EER Curves", fontsize=13, fontweight="bold")

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, color="#1565C0", linewidth=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1565C0")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # EER / DET
    ax = axes[1]
    ax.plot(fpr, fnr, color="#E65100", linewidth=2, label="DET Curve")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="FAR = FRR")
    if eer is not None:
        ax.scatter([eer], [eer], c="red", zorder=5, s=80,
                   label=f"EER = {eer*100:.2f}%\n(thresh = {eer_threshold:.4f})")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("False Rejection Rate (FRR)")
    ax.set_title("EER / DET Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    return _save(fig, "fig6_roc_eer.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────────────────

def fig_precision_recall(y_true, y_prob):
    """Precision-Recall curve with AP score."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#7B1FA2", linewidth=2,
            label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color="#7B1FA2")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, "fig7_precision_recall.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Score / Probability Distribution (Real vs Fake)
# ─────────────────────────────────────────────────────────────────────────────

def fig_score_distribution(y_true, y_prob):
    """Histogram of predicted probabilities separated by true class."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 40)

    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.65,
            color="#2196F3", label="Real", density=True)
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.65,
            color="#F44336", label="Fake", density=True)

    ax.set_xlabel("Predicted Probability (Fake)")
    ax.set_ylabel("Density")
    ax.set_title("Classifier Score Distribution", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, "fig8_score_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — t-SNE of GRU Embeddings  (paper Figure 5)
# ─────────────────────────────────────────────────────────────────────────────

def fig_tsne(dataset: DeepfakeAudioDataset,
             pretrained_encoder=None,
             max_samples: int = 500):
    """2-D t-SNE scatter of encoder embeddings coloured by class."""
    from torch.utils.data import DataLoader
    from dataset import collate_variable_length

    encoder = pretrained_encoder
    if encoder is None:
        encoder = GRUEmbedder().to(config.DEVICE)
    encoder.eval()

    loader = DataLoader(dataset, batch_size=32,
                        collate_fn=collate_variable_length,
                        shuffle=False, num_workers=0)

    all_emb, all_labels = [], []
    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(config.DEVICE)
            emb = encoder(specs)
            all_emb.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())
            if sum(len(e) for e in all_emb) >= max_samples:
                break

    emb_np  = np.concatenate(all_emb,   axis=0)[:max_samples]
    lab_np  = np.concatenate(all_labels, axis=0)[:max_samples]

    print(f"  [t-SNE] Fitting on {len(emb_np)} samples …")
    coords = TSNE(n_components=2, perplexity=30,
                  n_iter=1000, random_state=42).fit_transform(emb_np)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls, color, name in [(0, "#2196F3", "Real"), (1, "#F44336", "Fake")]:
        mask = lab_np == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=name, alpha=0.6, s=18, edgecolors="none")

    ax.set_title("t-SNE of GRU Encoder Embeddings", fontsize=13)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, "fig9_tsne_embeddings.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — PCA of GRU Embeddings (fast alternative to t-SNE)
# ─────────────────────────────────────────────────────────────────────────────

def fig_pca(dataset: DeepfakeAudioDataset,
            pretrained_encoder=None,
            max_samples: int = 1000):
    """2-D PCA of encoder embeddings."""
    from torch.utils.data import DataLoader
    from dataset import collate_variable_length

    encoder = pretrained_encoder
    if encoder is None:
        encoder = GRUEmbedder().to(config.DEVICE)
    encoder.eval()

    loader = DataLoader(dataset, batch_size=64,
                        collate_fn=collate_variable_length,
                        shuffle=False, num_workers=0)

    all_emb, all_labels = [], []
    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(config.DEVICE)
            emb = encoder(specs)
            all_emb.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())
            if sum(len(e) for e in all_emb) >= max_samples:
                break

    emb_np = np.concatenate(all_emb,   axis=0)[:max_samples]
    lab_np = np.concatenate(all_labels, axis=0)[:max_samples]

    pca    = PCA(n_components=2)
    coords = pca.fit_transform(emb_np)
    var    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls, color, name in [(0, "#2196F3", "Real"), (1, "#F44336", "Fake")]:
        mask = lab_np == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=name, alpha=0.6, s=18, edgecolors="none")

    ax.set_title("PCA of GRU Encoder Embeddings", fontsize=13)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, "fig10_pca_embeddings.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — Mel Frequency Band Mean Energy (Real vs Fake comparison)
# ─────────────────────────────────────────────────────────────────────────────

def fig_mean_mel_energy(dataset: DeepfakeAudioDataset, n_samples: int = 50):
    """Average energy per Mel band for real vs. fake audio."""
    real_mels, fake_mels = [], []

    for path, label in dataset.samples[:]:
        wav = load_audio(path)
        if wav is None:
            continue
        mel = librosa.feature.melspectrogram(
            y=wav, sr=config.SAMPLE_RATE,
            n_mels=config.N_MELS, n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        mean_energy = log_mel.mean(axis=1)   # (n_mels,)
        if label == 0:
            real_mels.append(mean_energy)
        else:
            fake_mels.append(mean_energy)
        if len(real_mels) >= n_samples and len(fake_mels) >= n_samples:
            break

    real_mean = np.stack(real_mels).mean(axis=0) if real_mels else np.zeros(config.N_MELS)
    fake_mean = np.stack(fake_mels).mean(axis=0) if fake_mels else np.zeros(config.N_MELS)

    freqs = librosa.mel_frequencies(n_mels=config.N_MELS,
                                     fmin=config.F_MIN, fmax=config.F_MAX)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, real_mean, color="#2196F3", linewidth=2, label="Real")
    ax.plot(freqs, fake_mean, color="#F44336", linewidth=2, label="Fake", linestyle="--")
    ax.fill_between(freqs, real_mean, fake_mean, alpha=0.1, color="purple")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz, log scale)")
    ax.set_ylabel("Mean Log-Mel Energy (dB)")
    ax.set_title("Average Mel-Band Energy: Real vs Fake", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, "fig11_mean_mel_energy.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 — Per-Epoch Metrics Summary Table (as figure)
# ─────────────────────────────────────────────────────────────────────────────

def fig_metrics_table(results: dict):
    """Render the final evaluation metrics as a clean table figure."""
    metrics = [
        ("Accuracy",   f"{results['accuracy']*100:.2f}%"),
        ("Precision",  f"{results['precision']*100:.2f}%"),
        ("Recall",     f"{results['recall']*100:.2f}%"),
        ("F1-Score",   f"{results['f1_score']*100:.2f}%"),
        ("AUC-ROC",    f"{results['auc_roc']:.4f}"),
        ("EER",        f"{results['eer']*100:.2f}%"),
        ("EER Thresh", f"{results['eer_threshold']:.4f}"),
    ]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.axis("off")
    table = ax.table(
        cellText=[[k, v] for k, v in metrics],
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 1.6)

    # Header styling
    for col in [0, 1]:
        table[0, col].set_facecolor("#1565C0")
        table[0, col].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for row in range(1, len(metrics) + 1):
        bg = "#E3F2FD" if row % 2 == 0 else "white"
        for col in [0, 1]:
            table[row, col].set_facecolor(bg)

    ax.set_title("Model Evaluation Results", fontsize=13,
                 fontweight="bold", pad=20)
    plt.tight_layout()
    return _save(fig, "fig12_metrics_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION — call this once to generate everything
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(
    model=None,
    history=None,
    dataset=None,
    results=None,
    y_true=None,
    y_pred=None,
    y_prob=None,
    pretrained_encoder=None,
):
    """
    Generate all paper figures. Pass None for any unavailable argument
    and the corresponding figure(s) will be skipped.
    """
    print("=" * 60)
    print("  Generating Paper Figures")
    print("=" * 60)

    paths = {}

    # Dataset plots (need dataset)
    if dataset is not None:
        print("\n[FIG 1] Dataset distribution …")
        paths["fig1"] = fig_dataset_distribution(dataset)

        print("\n[FIG 2] Sample waveforms …")
        try:
            paths["fig2"] = fig_sample_waveforms(dataset)
        except Exception as e:
            print(f"  [SKIP] {e}")

        print("\n[FIG 3] Mel spectrograms …")
        try:
            paths["fig3"] = fig_mel_spectrograms(dataset)
        except Exception as e:
            print(f"  [SKIP] {e}")

        print("\n[FIG 11] Mean Mel-band energy …")
        try:
            paths["fig11"] = fig_mean_mel_energy(dataset)
        except Exception as e:
            print(f"  [SKIP] {e}")

    # Training history plots
    if history is not None:
        print("\n[FIG 4] Training curves …")
        paths["fig4"] = fig_training_curves(history)

    # Evaluation plots (need y_true, y_pred, y_prob)
    if y_true is not None and y_pred is not None:
        print("\n[FIG 5] Confusion matrix …")
        paths["fig5"] = fig_confusion_matrix(y_true, y_pred)

    if y_true is not None and y_prob is not None:
        eer        = results.get("eer")        if results else None
        eer_thresh = results.get("eer_threshold") if results else None

        print("\n[FIG 6] ROC + EER curves …")
        paths["fig6"] = fig_roc_curve(y_true, y_prob, eer, eer_thresh)

        print("\n[FIG 7] Precision-Recall curve …")
        paths["fig7"] = fig_precision_recall(y_true, y_prob)

        print("\n[FIG 8] Score distribution …")
        paths["fig8"] = fig_score_distribution(y_true, y_prob)

    # Embedding plots (need dataset + optionally encoder)
    if dataset is not None:
        enc = pretrained_encoder
        if enc is None and model is not None:
            # Try to extract encoder from FineTunedModel
            if hasattr(model, "encoder"):
                enc = model.encoder

        print("\n[FIG 9] t-SNE of embeddings …")
        try:
            paths["fig9"] = fig_tsne(dataset, enc)
        except Exception as e:
            print(f"  [SKIP] {e}")

        print("\n[FIG 10] PCA of embeddings …")
        try:
            paths["fig10"] = fig_pca(dataset, enc)
        except Exception as e:
            print(f"  [SKIP] {e}")

    # Metrics table
    if results is not None:
        print("\n[FIG 12] Metrics table …")
        paths["fig12"] = fig_metrics_table(results)

    print("\n" + "=" * 60)
    print(f"  Done!  {len(paths)} figures saved to: {SAVE_DIR}/")
    print("=" * 60)
    return paths


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Run generate_all(...) from your notebook with the trained objects.")
