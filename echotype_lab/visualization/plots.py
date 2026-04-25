"""
echotype_lab.visualization.plots
==================================
Plotting utilities for keystroke audio analysis.

Functions
---------
- :func:`plot_waveform`        – time-domain waveform
- :func:`plot_spectrogram`     – mel-spectrogram
- :func:`plot_mfcc`            – MFCC heatmap
- :func:`plot_confusion_matrix` – annotated confusion matrix
- :func:`plot_class_distribution` – bar chart of class counts

All functions return the ``matplotlib.figure.Figure`` object so callers can
save or embed them freely.  Pass ``show=True`` to display interactively.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_figure(figsize=(10, 4)):
    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_waveform(
    samples: np.ndarray,
    sample_rate: int = 44_100,
    title: str = "Keystroke Waveform",
    *,
    show: bool = False,
    save_path: Optional[Path | str] = None,
):
    """Plot the time-domain waveform of a keystroke audio clip.

    Parameters
    ----------
    samples:
        1-D array of audio samples (int16 or float32).
    sample_rate:
        Sample rate in Hz.
    title:
        Plot title.
    show:
        If *True*, call ``plt.show()`` before returning.
    save_path:
        Optional path to save the figure (PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    samples = np.asarray(samples, dtype=np.float32)
    if samples.max() > 1.0:
        samples = samples / 32768.0

    fig, ax = _get_figure()
    t = np.linspace(0, len(samples) / sample_rate, num=len(samples))
    ax.plot(t, samples, color="steelblue", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_spectrogram(
    samples: np.ndarray,
    sample_rate: int = 44_100,
    title: str = "Mel Spectrogram",
    n_mels: int = 64,
    hop_length: int = 512,
    n_fft: int = 2_048,
    *,
    show: bool = False,
    save_path: Optional[Path | str] = None,
):
    """Plot a Mel-spectrogram of *samples*.

    Parameters
    ----------
    samples:
        1-D array of audio samples.
    sample_rate:
        Sample rate in Hz.
    title:
        Plot title.
    n_mels:
        Number of Mel filter banks.
    hop_length:
        STFT hop length.
    n_fft:
        FFT window size.
    show:
        Display interactively if *True*.
    save_path:
        Optional path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import librosa  # noqa: PLC0415
    import librosa.display  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    y = np.asarray(samples, dtype=np.float32)
    if y.max() > 1.0:
        y = y / 32768.0

    mel = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_mels=n_mels,
        hop_length=hop_length, n_fft=n_fft,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = _get_figure(figsize=(10, 5))
    img = librosa.display.specshow(
        mel_db, sr=sample_rate, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_mfcc(
    samples: np.ndarray,
    sample_rate: int = 44_100,
    n_mfcc: int = 13,
    hop_length: int = 512,
    n_fft: int = 2_048,
    title: str = "MFCC",
    *,
    show: bool = False,
    save_path: Optional[Path | str] = None,
):
    """Plot MFCC coefficients as a heatmap.

    Parameters
    ----------
    samples:
        1-D audio sample array.
    sample_rate:
        Sample rate in Hz.
    n_mfcc:
        Number of MFCC coefficients.
    hop_length:
        STFT hop length.
    n_fft:
        FFT window size.
    title:
        Plot title.
    show:
        Display interactively if *True*.
    save_path:
        Optional path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import librosa  # noqa: PLC0415
    import librosa.display  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    y = np.asarray(samples, dtype=np.float32)
    if y.max() > 1.0:
        y = y / 32768.0

    mfcc = librosa.feature.mfcc(
        y=y, sr=sample_rate, n_mfcc=n_mfcc,
        hop_length=hop_length, n_fft=n_fft,
    )

    fig, ax = _get_figure(figsize=(10, 4))
    img = librosa.display.specshow(
        mfcc, sr=sample_rate, hop_length=hop_length,
        x_axis="time", ax=ax,
    )
    fig.colorbar(img, ax=ax)
    ax.set_ylabel("MFCC coefficient")
    ax.set_title(title)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    *,
    normalize: bool = False,
    show: bool = False,
    save_path: Optional[Path | str] = None,
):
    """Plot an annotated confusion matrix.

    Parameters
    ----------
    cm:
        2-D integer confusion matrix of shape ``(n_classes, n_classes)``.
    labels:
        Class name labels for axis ticks.
    title:
        Plot title.
    normalize:
        If *True*, normalise each row to percentages.
    show:
        Display interactively if *True*.
    save_path:
        Optional path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums != 0, cm / row_sums, 0.0)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = ".0f"

    n = cm_plot.shape[0]
    figsize = (max(6, n), max(5, n - 1))
    fig, ax = _get_figure(figsize=figsize)

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels or list(range(n)),
        yticklabels=labels or list(range(n)),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_class_distribution(
    y: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Class Distribution",
    *,
    show: bool = False,
    save_path: Optional[Path | str] = None,
):
    """Plot a bar chart showing the number of samples per class.

    Parameters
    ----------
    y:
        1-D integer label array.
    labels:
        Class name labels for the x-axis.
    title:
        Plot title.
    show:
        Display interactively if *True*.
    save_path:
        Optional path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    y = np.asarray(y, dtype=np.int64)
    classes = sorted(set(y.tolist()))
    counts = [int((y == c).sum()) for c in classes]
    tick_labels = labels if labels else [str(c) for c in classes]

    fig, ax = _get_figure(figsize=(max(8, len(classes) * 0.6), 4))
    ax.bar(tick_labels, counts, color="steelblue", edgecolor="white")
    ax.set_xlabel("Key label")
    ax.set_ylabel("Sample count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _maybe_save(fig, save_path: Optional[Path | str]) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.debug("Figure saved to %s", save_path)
