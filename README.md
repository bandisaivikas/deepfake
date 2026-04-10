# Deepfake Audio Detection — SSL + BiGRU

> Self-supervised learning pipeline for deepfake audio detection using GRU-based embeddings and supervised BiGRU classification.

## Architecture
```
Raw Audio
    │
SSL Encoder (pretrained)  ← self-supervised feature extraction
    │
GRU Embedder              ← temporal sequence modeling
    │
FineTuned BiGRU Classifier ← supervised deepfake detection
    │
Real / Fake
```

## Results

| Metric | Value |
|---|---|
| Task | Binary deepfake audio detection |
| Architecture | SSL + BiGRU |
| Figures generated | 11 |

## Files

| File | Description |
|---|---|
| `main.ipynb` | Full pipeline notebook |
| `train.py` | Training script |
| `models.py` | Model definitions |
| `inference.py` | Inference script |
| `config.py` | Hyperparameters and paths |
| `dataset.py` | Data loading |
| `embeddings.py` | Embedding extraction |
| `losses.py` | Loss functions |
| `paper_figures.py` | Figure generation |
| `visualize.py` | Visualization utilities |

## Setup
```bash
git clone https://github.com/bandisaivikas/deepfake.git
cd deepfake
pip install -r requirements.txt
```

## Author

**Saivikas bandi,Navneeth Kundrapu,Chaitanya krishna
** · IIIT Naya Raipur · B.Tech Data Science & AI · Batch 2027
