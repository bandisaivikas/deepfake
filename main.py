#!/usr/bin/env python3
# main.py - Full pipeline entry point

"""
Deepfake Audio Detection – Full Pipeline
=========================================
Based on: "Deepfake Audio Detection - Using SSL"
Architecture: Mel-spectrogram + Contrastive Pre-training (NT-Xent) + BiGRU Classifier

Usage
-----
  # Full pipeline (pre-train + supervised training + evaluate)
  python main.py

  # Skip contrastive pre-training, train classifier from scratch
  python main.py --skip-pretrain

  # Only generate embeddings from saved spectrograms
  python main.py --embeddings-only

  # Only evaluate a saved checkpoint
  python main.py --eval-only

  # Adjust data directory
  python main.py --data-dir /path/to/dataset

Dataset Layout Expected
-----------------------
  data/
    real/    ← .wav / .mp3 / .flac / .ogg files of genuine speech
    fake/    ← .wav / .mp3 / .flac / .ogg files of AI-generated speech
"""

import os
import sys
import argparse
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import config
from dataset import (DeepfakeAudioDataset, extract_log_mel_spectrograms,
                     collate_variable_length)
from embeddings import generate_rnn_embeddings
from models import BiGRUClassifier, count_parameters
from train import pretrain_contrastive, train_classifier, evaluate, print_results
from visualize import (plot_training_curves, plot_tsne,
                       plot_confusion_matrix, plot_roc_eer)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Deepfake Audio Detection Pipeline")
    p.add_argument("--data-dir",       default=config.DATA_DIR, help="Root data directory")
    p.add_argument("--epochs",         type=int, default=config.EPOCHS)
    p.add_argument("--pretrain-epochs",type=int, default=10)
    p.add_argument("--batch-size",     type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=config.LEARNING_RATE)
    p.add_argument("--skip-pretrain",  action="store_true", help="Skip Phase 1")
    p.add_argument("--eval-only",      action="store_true", help="Evaluate saved model")
    p.add_argument("--embeddings-only",action="store_true", help="Only generate embeddings")
    p.add_argument("--model",          default=config.BEST_MODEL_PATH)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: check dataset exists
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset(real_dir, fake_dir):
    missing = []
    if not os.path.isdir(real_dir): missing.append(real_dir)
    if not os.path.isdir(fake_dir): missing.append(fake_dir)
    if missing:
        print("\n[ERROR] Dataset directories not found:")
        for d in missing:
            print(f"  → {d}")
        print("\nPlease create the following layout:")
        print("  data/real/   ← genuine speech files (.wav/.mp3/.flac/.ogg)")
        print("  data/fake/   ← deepfake speech files")
        print("\nYou can point to a different root with --data-dir <path>")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Override config with CLI args
    config.DATA_DIR      = args.data_dir
    config.REAL_DIR      = os.path.join(args.data_dir, "real")
    config.FAKE_DIR      = os.path.join(args.data_dir, "fake")
    config.EPOCHS        = args.epochs
    config.BATCH_SIZE    = args.batch_size
    config.LEARNING_RATE = args.lr

    print("="*60)
    print("  Deepfake Audio Detection – Using SSL")
    print("="*60)
    print(f"  Device : {config.DEVICE}")
    print(f"  Data   : {config.DATA_DIR}")
    print(f"  Epochs : {config.EPOCHS}")
    print(f"  Batch  : {config.BATCH_SIZE}")
    print(f"  LR     : {config.LEARNING_RATE}")
    print()

    check_dataset(config.REAL_DIR, config.FAKE_DIR)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = DeepfakeAudioDataset(config.REAL_DIR, config.FAKE_DIR)
    labels  = [s[1] for s in dataset.samples]
    n_real  = labels.count(0)
    n_fake  = labels.count(1)
    print(f"Dataset: {len(dataset.samples)} samples  |  Real: {n_real}  Fake: {n_fake}")

    if len(dataset) == 0:
        print("[ERROR] No audio files found in dataset.")
        sys.exit(1)

    # ── Embeddings-only mode ──────────────────────────────────────────────────
    if args.embeddings_only:
        print("\nGenerating log-Mel spectrograms …")
        specs, fnames = extract_log_mel_spectrograms(config.DATA_DIR)
        generate_rnn_embeddings(specs, fnames)
        return

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        model = BiGRUClassifier().to(config.DEVICE)
        state = torch.load(args.model, map_location=config.DEVICE)
        model.load_state_dict(state)
        print(f"Loaded model from {args.model}")

        _, val_idx = train_test_split(
            list(range(len(dataset))), test_size=config.VAL_SPLIT,
            stratify=labels, random_state=config.RANDOM_STATE)
        val_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            sampler=SubsetRandomSampler(val_idx),
            collate_fn=collate_variable_length)
        results, y_true, y_pred, y_prob = evaluate(model, val_loader)
        print_results(results)
        return

    # ── Phase 1: Contrastive pre-training ────────────────────────────────────
    pretrained_encoder = None
    if not args.skip_pretrain:
        contra_model = pretrain_contrastive(dataset, epochs=args.pretrain_epochs)
        pretrained_encoder = contra_model.encoder
    else:
        print("\n[Phase 1 skipped] Training classifier from scratch.")

    # ── Phase 2: Supervised classification ───────────────────────────────────
    model, history = train_classifier(
        dataset,
        pretrained_encoder=pretrained_encoder,
        epochs=config.EPOCHS
    )

    # ── Validation set evaluation ─────────────────────────────────────────────
    _, val_idx = train_test_split(
        list(range(len(dataset))), test_size=config.VAL_SPLIT,
        stratify=labels, random_state=config.RANDOM_STATE)
    val_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=SubsetRandomSampler(val_idx),
        collate_fn=collate_variable_length)

    # Load the best checkpoint for final metrics
    best_model = BiGRUClassifier().to(config.DEVICE)
    best_model.load_state_dict(torch.load(config.BEST_MODEL_PATH,
                                           map_location=config.DEVICE))
    results, y_true, y_pred, y_prob = evaluate(best_model, val_loader)
    print_results(results)

    # ── Generate and save embeddings ──────────────────────────────────────────
    print("\nGenerating spectrogram + RNN embeddings …")
    specs, fnames = extract_log_mel_spectrograms(config.DATA_DIR)
    emb_matrix, _ = generate_rnn_embeddings(specs, fnames)

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_training_curves(history)

    # t-SNE on validation embeddings
    y_true_arr = np.array(y_true)
    # Build embedding matrix for val samples only
    val_embs = emb_matrix[val_idx[:len(y_true_arr)]] \
        if len(emb_matrix) >= len(val_idx) else emb_matrix[:len(y_true_arr)]
    if len(val_embs) == len(y_true_arr):
        plot_tsne(val_embs, y_true_arr)

    plot_confusion_matrix(np.array(y_true), np.array(y_pred))
    plot_roc_eer(np.array(y_true), np.array(y_prob),
                 results["eer"], results["eer_threshold"])

    print("\n" + "="*60)
    print("  Pipeline complete.")
    print(f"  Best model   → {config.BEST_MODEL_PATH}")
    print(f"  Plots        → artifacts/plots/")
    print(f"  Embeddings   → {config.EMBEDDINGS_NPY}")
    print("="*60)


if __name__ == "__main__":
    main()
