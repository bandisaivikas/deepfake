# train.py - Two-phase training pipeline

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, roc_auc_score)

import config
from dataset import (DeepfakeAudioDataset, ContrastiveAudioDataset,
                     collate_variable_length)
from models import ContrastiveModel, BiGRUClassifier, FineTunedModel, count_parameters
from losses import NTXentLoss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_train_val_indices(dataset: DeepfakeAudioDataset):
    labels = [s[1] for s in dataset.samples]
    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.VAL_SPLIT,
        stratify=labels,
        random_state=config.RANDOM_STATE
    )
    return train_idx, val_idx


def make_loaders(dataset: DeepfakeAudioDataset):
    train_idx, val_idx = get_train_val_indices(dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=SubsetRandomSampler(train_idx),
        collate_fn=collate_variable_length,
        num_workers=0,
        drop_last=True        # avoid batch-size-1 when dataset % BATCH_SIZE == 1
    )
    val_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=SubsetRandomSampler(val_idx),
        collate_fn=collate_variable_length,
        num_workers=0
    )
    return train_loader, val_loader, train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – Self-Supervised Contrastive Pre-training
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_contrastive(base_dataset: DeepfakeAudioDataset,
                         epochs: int = 10) -> ContrastiveModel:
    """
    Train the ContrastiveModel (GRUEmbedder + ProjectionHead) with NT-Xent loss
    on augmented pairs of audio samples.  No labels are used.
    """
    print("\n" + "="*60)
    print("Phase 1 – Contrastive Pre-training")
    print("="*60)

    contra_ds = ContrastiveAudioDataset(base_dataset)

    def _collate(batch):
        v1s, v2s, labels = zip(*batch)
        max_t = max(v.shape[-1] for v in (*v1s, *v2s))
        def _pad(views):
            out = torch.zeros(len(views), config.N_MELS, max_t)
            for i, v in enumerate(views):
                out[i, :, :v.shape[-1]] = v
            return out
        return _pad(v1s), _pad(v2s), torch.stack(labels)

    loader = DataLoader(contra_ds, batch_size=config.BATCH_SIZE,
                        shuffle=True, collate_fn=_collate, num_workers=0)

    model     = ContrastiveModel().to(config.DEVICE)
    criterion = NTXentLoss(temperature=config.TEMPERATURE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.SCHEDULER_STEP, gamma=config.SCHEDULER_GAMMA)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for v1, v2, _ in loader:
            v1, v2 = v1.to(config.DEVICE), v2.to(config.DEVICE)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / max(len(loader), 1)
        print(f"  Epoch [{epoch:02d}/{epochs}]  Loss: {avg:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    os.makedirs(os.path.dirname(config.PRETRAIN_PATH), exist_ok=True)
    torch.save(model.encoder.state_dict(), config.PRETRAIN_PATH)
    print(f"  Encoder saved → {config.PRETRAIN_PATH}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – Supervised Fine-tuning  (or train from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def _one_epoch_train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(config.DEVICE), labels.to(config.DEVICE)
        logits = model(specs)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def _one_epoch_val(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []
    for specs, labels in loader:
        specs, labels = specs.to(config.DEVICE), labels.to(config.DEVICE)
        logits = model(specs)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    acc = correct / max(total, 1)
    return total_loss / max(len(loader), 1), acc, all_labels, all_probs


def train_classifier(dataset: DeepfakeAudioDataset,
                     pretrained_encoder=None,
                     epochs: int = config.EPOCHS) -> nn.Module:
    """
    Train (or fine-tune) the BiGRU classification model.

    If pretrained_encoder is provided, wraps it in FineTunedModel.
    Otherwise trains BiGRUClassifier from scratch.
    """
    print("\n" + "="*60)
    print("Phase 2 – Supervised Classification")
    print("="*60)

    train_loader, val_loader, _, _ = make_loaders(dataset)

    if pretrained_encoder is not None:
        model = FineTunedModel(pretrained_encoder).to(config.DEVICE)
    else:
        model = BiGRUClassifier().to(config.DEVICE)

    print(f"  Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.LEARNING_RATE,
                                 betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.SCHEDULER_STEP, gamma=config.SCHEDULER_GAMMA)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _one_epoch_train(model, train_loader, criterion, optimizer)
        va_loss, va_acc, va_labels, va_probs = _one_epoch_val(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        elapsed = time.time() - t0
        print(f"  Epoch [{epoch:02d}/{epochs}]  "
              f"TrLoss: {tr_loss:.4f}  TrAcc: {tr_acc:.4f}  "
              f"VaLoss: {va_loss:.4f}  VaAcc: {va_acc:.4f}  "
              f"({elapsed:.1f}s)")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            os.makedirs(os.path.dirname(config.BEST_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"    ✓ Best model saved  (val_acc={best_val_acc:.4f})")

    print(f"\n  Best validation accuracy: {best_val_acc:.4f}")

    # ── Persist history so it survives kernel restarts ─────────────────────
    import json
    history_path = os.path.join(os.path.dirname(config.BEST_MODEL_PATH),
                                "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"  History saved → {history_path}")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    """
    Full evaluation: accuracy, precision, recall, F1, AUC-ROC, EER.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for specs, labels in loader:
        specs, labels = specs.to(config.DEVICE), labels.to(config.DEVICE)
        logits = model(specs)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    acc  = (all_labels == all_preds).mean()
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    auc  = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else float("nan")
    eer, eer_threshold = compute_eer(all_labels, all_probs)

    results = {
        "accuracy":      acc,
        "precision":     prec,
        "recall":        rec,
        "f1_score":      f1,
        "auc_roc":       auc,
        "eer":           eer,
        "eer_threshold": eer_threshold
    }
    return results, all_labels, all_preds, all_probs


def compute_eer(labels: np.ndarray, scores: np.ndarray):
    """
    Compute Equal Error Rate (EER) and the score threshold where FAR ≈ FRR.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def print_results(results: dict):
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"  Accuracy      : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision     : {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall        : {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1-Score      : {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    print(f"  AUC-ROC       : {results['auc_roc']:.4f}")
    print(f"  EER           : {results['eer']:.4f} ({results['eer']*100:.2f}%)")
    print(f"  EER Threshold : {results['eer_threshold']:.4f}")
    print("="*60)
