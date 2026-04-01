# models.py - All neural network architectures described in the paper

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2 – GenerateRNNEmbeddings (feature extraction GRU)
# ─────────────────────────────────────────────────────────────────────────────

class GRUEmbedder(nn.Module):
    """
    Unidirectional single-layer GRU that encodes a Mel spectrogram sequence
    into a fixed-length 128-d vector (last hidden state).

    Input:  (batch, n_mels, time)  – i.e. spectrogram
    Output: (batch, hidden_size)
    """
    def __init__(self,
                 input_size:  int = config.N_MELS,
                 hidden_size: int = config.GRU_HIDDEN_SIZE,
                 num_layers:  int = config.GRU_NUM_LAYERS):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_mels, time) → transpose → (batch, time, n_mels)
        x = x.permute(0, 2, 1)
        _, hidden = self.gru(x)          # hidden: (num_layers, batch, hidden)
        return hidden[-1]                # last layer: (batch, hidden_size)


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive Learning – Encoder + Projection Head  (Section 3.3, Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head that maps encoder outputs to a space
    where NT-Xent contrastive loss is applied.
    """
    def __init__(self,
                 in_dim:  int = config.GRU_HIDDEN_SIZE,
                 out_dim: int = config.PROJECTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveModel(nn.Module):
    """
    Encoder f(·) + Projection head g(·) used in self-supervised Phase 1.
    h = f(x)   (encoder embedding)
    z = g(h)   (projection for loss computation)
    """
    def __init__(self):
        super().__init__()
        self.encoder   = GRUEmbedder()
        self.projector = ProjectionHead()

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


# ─────────────────────────────────────────────────────────────────────────────
# Classification Model – Bidirectional GRU  (Section 5.3)
# ─────────────────────────────────────────────────────────────────────────────

class BiGRUClassifier(nn.Module):
    """
    Two-layer Bidirectional GRU followed by a linear classifier.

    Input shape:  (batch, n_mels, time)
    Output shape: (batch, num_classes)

    ~67 k parameters as described in the paper.
    """
    def __init__(self,
                 input_size:  int = config.N_MELS,
                 hidden_size: int = config.BIGRU_HIDDEN,
                 num_layers:  int = config.BIGRU_LAYERS,
                 num_classes: int = config.NUM_CLASSES,
                 dropout:     float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_mels, time) → (batch, time, n_mels)
        x = x.permute(0, 2, 1)
        _, hidden = self.gru(x)
        # Concat last layer forward & backward hidden states
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, 2*hidden)
        out = self.norm(out)
        out = self.drop(out)
        return self.fc(out)                               # (batch, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning wrapper  (Section 3.3, Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

class FineTunedModel(nn.Module):
    """
    Pretrained encoder + MLP classifier used in supervised fine-tuning.
    The projection head is discarded; a new linear head is attached.
    """
    def __init__(self,
                 encoder:     nn.Module,
                 in_features: int = config.GRU_HIDDEN_SIZE,
                 num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
