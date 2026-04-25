"""
echotype_lab.features.mfcc
===========================
Extract Mel-frequency Cepstral Coefficients (MFCCs) from keystroke audio
clips using ``librosa``.

The :func:`extract_mfcc` function returns a 1-D feature vector suitable for
use as input to a classifier.  The :func:`extract_mfcc_batch` helper
processes a directory of WAV files and returns arrays ready for model
training.

Usage example
-------------
>>> from echotype_lab.features.mfcc import extract_mfcc, extract_mfcc_batch
>>> feats = extract_mfcc("data/raw/a/a_0001.wav")
>>> feats.shape
(78,)   # n_mfcc=13, statistics: mean+std+delta_mean+delta_std+delta2_mean+delta2_std → 13*6
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_MFCC: int = 13
DEFAULT_SAMPLE_RATE: int = 44_100
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_N_FFT: int = 2_048


def extract_mfcc(
    audio_path: Path | str,
    n_mfcc: int = DEFAULT_N_MFCC,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
) -> np.ndarray:
    """Extract a fixed-size MFCC feature vector from a WAV file.

    The vector is formed by computing MFCCs and their first and second
    temporal derivatives (deltas), then summarising each coefficient track
    with its mean and standard deviation.  The final vector has length
    ``n_mfcc * 6``.

    Parameters
    ----------
    audio_path:
        Path to a mono WAV file.
    n_mfcc:
        Number of MFCC coefficients to compute.
    sample_rate:
        Expected sample rate of the audio.  librosa resamples if needed.
    hop_length:
        STFT hop length in samples.
    n_fft:
        FFT window size in samples.

    Returns
    -------
    numpy.ndarray
        1-D float32 array of length ``n_mfcc * 6``.
    """
    import librosa  # noqa: PLC0415

    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                 hop_length=hop_length, n_fft=n_fft)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    parts = [
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        delta.std(axis=1),
        delta2.mean(axis=1),
        delta2.std(axis=1),
    ]
    return np.concatenate(parts).astype(np.float32)


def extract_mfcc_from_array(
    samples: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mfcc: int = DEFAULT_N_MFCC,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
) -> np.ndarray:
    """Like :func:`extract_mfcc` but accepts a raw sample array.

    Parameters
    ----------
    samples:
        1-D array of audio samples (float32 or int16).
    sample_rate:
        Sample rate of *samples*.

    Returns
    -------
    numpy.ndarray
        1-D float32 feature vector of length ``n_mfcc * 6``.
    """
    import librosa  # noqa: PLC0415

    y = samples.astype(np.float32)
    if y.max() > 1.0:
        y = y / 32768.0  # normalise int16 range to [-1, 1]

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc,
                                 hop_length=hop_length, n_fft=n_fft)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    parts = [
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        delta.std(axis=1),
        delta2.mean(axis=1),
        delta2.std(axis=1),
    ]
    return np.concatenate(parts).astype(np.float32)


def extract_mfcc_batch(
    data_dir: Path | str,
    n_mfcc: int = DEFAULT_N_MFCC,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Process an entire labelled dataset directory and return (X, y, labels).

    Expected directory layout::

        data_dir/
            a/
                a_0001.wav
                a_0002.wav
            b/
                b_0001.wav
            ...

    Parameters
    ----------
    data_dir:
        Root directory containing one sub-directory per key label.

    Returns
    -------
    X : numpy.ndarray
        2-D float32 array of shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        1-D int64 label array of shape ``(n_samples,)``.
    labels : list[str]
        Sorted list of unique label names (index corresponds to *y* values).
    """
    data_dir = Path(data_dir)
    label_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    labels = [p.name for p in label_dirs]

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for label_idx, label_dir in enumerate(label_dirs):
        wav_files = sorted(label_dir.glob("*.wav"))
        if not wav_files:
            logger.warning("No WAV files found in %s – skipping", label_dir)
            continue
        for wav_path in wav_files:
            try:
                feats = extract_mfcc(wav_path, n_mfcc=n_mfcc,
                                     sample_rate=sample_rate,
                                     hop_length=hop_length, n_fft=n_fft)
                X_rows.append(feats)
                y_rows.append(label_idx)
            except Exception:
                logger.exception("Failed to process %s – skipping", wav_path)

    if not X_rows:
        raise ValueError(f"No valid audio files found under {data_dir}")

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.int64)
    logger.info("Loaded %d samples, %d classes", len(X_rows), len(labels))
    return X, y, labels
