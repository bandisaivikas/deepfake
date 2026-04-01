# config.py - Central configuration for Deepfake Audio Detection

import torch

# ─── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16_000          # Hz
DURATION         = 3.0             # seconds
NUM_SAMPLES      = int(SAMPLE_RATE * DURATION)   # 48 000 samples
MIN_SAMPLES      = 1_600           # skip clips shorter than 0.1 s

# ─── Mel Spectrogram ──────────────────────────────────────────────────────────
N_MELS           = 128
N_FFT            = 1_024
HOP_LENGTH       = 512
F_MIN            = 0.0
F_MAX            = 8_000.0

# ─── GRU Embedding ────────────────────────────────────────────────────────────
GRU_HIDDEN_SIZE  = 128             # Phase 1 / feature extraction GRU
GRU_NUM_LAYERS   = 1

# ─── Classification Model (BiGRU) ─────────────────────────────────────────────
BIGRU_HIDDEN     = 64              # units per direction
BIGRU_LAYERS     = 2
NUM_CLASSES      = 2               # real / fake

# ─── Contrastive Pre-Training ─────────────────────────────────────────────────
PROJECTION_DIM   = 128
TEMPERATURE      = 0.07

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE       = 16
EPOCHS           = 20
LEARNING_RATE    = 5e-4
SCHEDULER_STEP   = 5
SCHEDULER_GAMMA  = 0.8
VAL_SPLIT        = 0.2
RANDOM_STATE     = 42

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR         = "data"          # root: data/real/  &  data/fake/
REAL_DIR         = f"{DATA_DIR}/real"
FAKE_DIR         = f"{DATA_DIR}/fake"

SPECTROGRAM_PT   = "artifacts/spectrograms.pt"
EMBEDDINGS_PT    = "artifacts/embeddings.pt"
EMBEDDINGS_NPY   = "artifacts/embeddings.npy"
EMBEDDINGS_CSV   = "artifacts/embeddings.csv"
BEST_MODEL_PATH  = "artifacts/best_audio_model.pth"
PRETRAIN_PATH    = "artifacts/pretrained_encoder.pth"

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
