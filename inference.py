# inference.py - Run the trained model on new audio files

import os
import sys
import argparse
import numpy as np
import torch
from typing import List, Tuple

import config
from dataset import load_audio, wav_to_log_mel
from models import BiGRUClassifier, FineTunedModel, GRUEmbedder


# ─────────────────────────────────────────────────────────────────────────────
# Load saved model
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(checkpoint: str = config.BEST_MODEL_PATH) -> torch.nn.Module:
    model = BiGRUClassifier().to(config.DEVICE)
    state = torch.load(checkpoint, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Predict a single file
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_file(path: str, model: torch.nn.Module) -> dict:
    """
    Returns dict with keys: filename, label, confidence, is_fake
    """
    wav = load_audio(path)
    if wav is None:
        return {"filename": os.path.basename(path), "error": "Could not load audio"}

    log_mel = wav_to_log_mel(wav)                                # (n_mels, time)
    tensor  = torch.tensor(log_mel).unsqueeze(0).to(config.DEVICE)  # (1, n_mels, time)

    logits = model(tensor)                                       # (1, 2)
    probs  = torch.softmax(logits, dim=1).squeeze()              # (2,)
    pred   = logits.argmax(dim=1).item()                         # 0=real, 1=fake

    return {
        "filename":   os.path.basename(path),
        "label":      "fake" if pred == 1 else "real",
        "is_fake":    pred == 1,
        "prob_real":  float(probs[0]),
        "prob_fake":  float(probs[1]),
        "confidence": float(probs[pred])
    }


@torch.no_grad()
def predict_batch(paths: List[str], model: torch.nn.Module) -> List[dict]:
    results = []
    for path in paths:
        result = predict_file(path, model)
        results.append(result)
        flag = "🔴 FAKE" if result.get("is_fake") else "🟢 REAL"
        print(f"  {flag}  ({result.get('confidence', 0):.3f})  {result.get('filename', path)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Inference")
    parser.add_argument("inputs", nargs="+", help="Audio file(s) or folder path")
    parser.add_argument("--model", default=config.BEST_MODEL_PATH,
                        help="Path to trained model checkpoint")
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model = load_classifier(args.model)

    # Collect all audio paths
    audio_paths: List[str] = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            for fname in sorted(os.listdir(inp)):
                if os.path.splitext(fname)[1].lower() in {".wav", ".mp3", ".flac", ".ogg"}:
                    audio_paths.append(os.path.join(inp, fname))
        elif os.path.isfile(inp):
            audio_paths.append(inp)
        else:
            print(f"Warning: {inp} not found, skipping.")

    if not audio_paths:
        print("No audio files found.")
        sys.exit(1)

    print(f"\nProcessing {len(audio_paths)} file(s) …\n")
    results = predict_batch(audio_paths, model)

    # Summary
    fakes = sum(1 for r in results if r.get("is_fake"))
    reals = len(results) - fakes
    print(f"\nSummary: {reals} real | {fakes} fake | {len(results)} total")
    return results


if __name__ == "__main__":
    main()
