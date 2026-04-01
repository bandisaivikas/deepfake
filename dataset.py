# dataset.py - Audio loading, preprocessing, and Dataset classes

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from typing import List, Tuple, Optional

import config


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def load_audio(path: str,
               target_sr: int = config.SAMPLE_RATE,
               num_samples: int = config.NUM_SAMPLES) -> Optional[np.ndarray]:
    """
    Load an audio file, resample to target_sr, convert to mono,
    and pad/trim to exactly num_samples.

    Returns None if the file is too short (< MIN_SAMPLES).
    """
    try:
        wav, _ = librosa.load(path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"[LOAD ERROR] {path}: {e}")
        return None

    if len(wav) < config.MIN_SAMPLES:
        print(f"[SKIPPED SHORT] {os.path.basename(path)}")
        return None

    # Pad or trim to fixed length
    if len(wav) < num_samples:
        wav = np.pad(wav, (0, num_samples - len(wav)))
    else:
        wav = wav[:num_samples]

    return wav.astype(np.float32)


def wav_to_log_mel(wav: np.ndarray,
                   sr: int = config.SAMPLE_RATE,
                   n_mels: int = config.N_MELS,
                   n_fft: int = config.N_FFT,
                   hop_length: int = config.HOP_LENGTH,
                   f_min: float = config.F_MIN,
                   f_max: float = config.F_MAX) -> np.ndarray:
    """
    Convert a waveform to a log-power Mel spectrogram and apply
    z-score normalisation.

    Output shape: (n_mels, time_frames)
    """
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=f_min, fmax=f_max,
        window="hann"
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # z-score normalisation per sample
    mean, std = log_mel.mean(), log_mel.std()
    if std > 1e-6:
        log_mel = (log_mel - mean) / std
    return log_mel.astype(np.float32)


def extract_statistical_features(log_mel: np.ndarray) -> np.ndarray:
    """
    Extract mean, std, skewness, and kurtosis across the time axis.
    Returns a (4 * n_mels,) vector.
    """
    from scipy.stats import skew, kurtosis
    mean_  = log_mel.mean(axis=1)
    std_   = log_mel.std(axis=1)
    skew_  = skew(log_mel, axis=1)
    kurt_  = kurtosis(log_mel, axis=1)
    return np.concatenate([mean_, std_, skew_, kurt_]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1 – ExtractLogMelSpectrograms
# ─────────────────────────────────────────────────────────────────────────────

def extract_log_mel_spectrograms(
    folder_path: str,
    target_sr: int = config.SAMPLE_RATE,
    n_mels: int = config.N_MELS,
    save_path: str = config.SPECTROGRAM_PT
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Implements Algorithm 1 from the paper.
    Walks folder_path, loads every supported audio file,
    converts to log-Mel spectrogram, saves to a .pt file.
    """
    spectrograms: List[np.ndarray] = []
    filenames:    List[str]        = []
    total_files   = 0

    for root, _, files in os.walk(folder_path):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in AUDIO_EXTENSIONS:
                continue
            total_files += 1
            fpath = os.path.join(root, fname)

            try:
                wav = load_audio(fpath, target_sr=target_sr)
                if wav is None:
                    continue
                log_mel = wav_to_log_mel(wav, sr=target_sr, n_mels=n_mels)
                spectrograms.append(log_mel)
                filenames.append(fname)
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")
                continue

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"spectrograms": spectrograms, "filenames": filenames}, save_path)
    print(f"[ExtractLogMelSpectrograms] Extracted: {len(spectrograms)}/{total_files}")
    return spectrograms, filenames


# ─────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeAudioDataset(Dataset):
    """
    Reads real/ and fake/ folders, returns (log_mel_tensor, label).
    label: 0 = real, 1 = fake
    """

    def __init__(self,
                 real_dir: str = config.REAL_DIR,
                 fake_dir: str = config.FAKE_DIR,
                 indices: Optional[List[int]] = None):
        self.samples: List[Tuple[str, int]] = []

        for fdir, label in [(real_dir, 0), (fake_dir, 1)]:
            if not os.path.isdir(fdir):
                continue
            for fname in sorted(os.listdir(fdir)):
                if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                    self.samples.append((os.path.join(fdir, fname), label))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        wav = load_audio(path)
        if wav is None:
            # Return a zero spectrogram for broken files
            log_mel = np.zeros((config.N_MELS, config.NUM_SAMPLES // config.HOP_LENGTH + 1),
                               dtype=np.float32)
        else:
            log_mel = wav_to_log_mel(wav)

        # shape → (1, n_mels, time) for CNN or (n_mels, time) for GRU
        return torch.tensor(log_mel), torch.tensor(label, dtype=torch.long)


class ContrastiveAudioDataset(Dataset):
    """
    Returns two augmented views of the same audio for contrastive pre-training.
    """

    def __init__(self, base_dataset: DeepfakeAudioDataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def _augment(self, wav: np.ndarray) -> np.ndarray:
        """Simple augmentations: time-masking, noise, pitch shift."""
        aug = wav.copy()
        # Gaussian noise
        aug += np.random.normal(0, 0.005, size=aug.shape).astype(np.float32)
        # Random time shift (up to 1600 samples / 0.1 s)
        shift = np.random.randint(0, config.MIN_SAMPLES)
        aug = np.roll(aug, shift)
        return aug

    def __getitem__(self, idx: int):
        path, label = self.base.samples[idx]
        wav = load_audio(path)
        if wav is None:
            wav = np.zeros(config.NUM_SAMPLES, dtype=np.float32)

        view1 = torch.tensor(wav_to_log_mel(self._augment(wav)))
        view2 = torch.tensor(wav_to_log_mel(self._augment(wav)))
        return view1, view2, torch.tensor(label, dtype=torch.long)


def collate_variable_length(batch):
    """
    Pad a batch of (log_mel, label) where time dimension may vary.
    Returns (padded_tensor, labels).
    """
    specs, labels = zip(*batch)
    max_t = max(s.shape[-1] for s in specs)
    padded = torch.zeros(len(specs), config.N_MELS, max_t)
    for i, s in enumerate(specs):
        t = s.shape[-1]
        padded[i, :, :t] = s
    return padded, torch.stack(labels)
