# embeddings.py - Algorithm 2: GenerateRNNEmbeddings

import os
import numpy as np
import torch
import pandas as pd
from typing import List, Tuple

import config
from models import GRUEmbedder


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2 – GenerateRNNEmbeddings
# ─────────────────────────────────────────────────────────────────────────────

def generate_rnn_embeddings(
    spectrograms:  List[np.ndarray],
    filenames:     List[str],
    hidden_size:   int = config.GRU_HIDDEN_SIZE,
    save_pt_path:  str = config.EMBEDDINGS_PT,
    save_npy_path: str = config.EMBEDDINGS_NPY,
    save_csv_path: str = config.EMBEDDINGS_CSV
) -> Tuple[np.ndarray, List[str]]:
    """
    Implements Algorithm 2 from the paper.

    For each log-Mel spectrogram:
      1. Transpose (n_mels, time) → (time, n_mels)
      2. Add batch dimension → (1, time, n_mels)
      3. Run through GRU, take last hidden state as embedding
      4. Collect all embeddings into a matrix

    Returns:
        all_embeddings : np.ndarray of shape (N, hidden_size)
        filenames      : corresponding list of filenames
    """
    model = GRUEmbedder(
        input_size=config.N_MELS,
        hidden_size=hidden_size,
        num_layers=config.GRU_NUM_LAYERS
    ).to(config.DEVICE)
    model.eval()

    embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for spec in spectrograms:
            # spec: (n_mels, time)
            t = torch.tensor(spec).to(config.DEVICE)         # (n_mels, time)
            t = t.unsqueeze(0)                                # (1, n_mels, time)
            emb = model(t)                                    # (1, hidden_size)
            embeddings.append(emb.cpu())

    all_embeddings = torch.cat(embeddings, dim=0)            # (N, hidden_size)
    emb_np = all_embeddings.numpy()

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_pt_path), exist_ok=True)
    torch.save({"embeddings": all_embeddings, "filenames": filenames}, save_pt_path)
    np.save(save_npy_path, emb_np)

    df = pd.DataFrame(emb_np)
    df.insert(0, "filename", filenames)
    df.to_csv(save_csv_path, index=False)

    print(f"[GenerateRNNEmbeddings] Saved {len(embeddings)} embeddings")
    print(f"  → {save_pt_path}")
    print(f"  → {save_npy_path}")
    print(f"  → {save_csv_path}")

    return emb_np, filenames


def load_embeddings(pt_path: str = config.EMBEDDINGS_PT):
    """Load previously saved embeddings and filenames."""
    data = torch.load(pt_path, map_location="cpu")
    return data["embeddings"].numpy(), data["filenames"]
