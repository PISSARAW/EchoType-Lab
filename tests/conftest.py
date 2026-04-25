"""Shared pytest fixtures for EchoType Lab tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def sample_audio_int16():
    """Return a synthetic 0.5 s, 44100 Hz mono int16 audio clip."""
    rng = np.random.default_rng(0)
    return rng.integers(-32768, 32767, size=22_050, dtype=np.int16)


@pytest.fixture()
def sample_audio_float32():
    """Return a synthetic 0.5 s float32 audio clip normalised to [-1, 1]."""
    rng = np.random.default_rng(1)
    return rng.uniform(-1.0, 1.0, size=22_050).astype(np.float32)


@pytest.fixture()
def simple_dataset():
    """Return a tiny (X, y, labels) dataset for classifier tests."""
    rng = np.random.default_rng(42)
    n_classes = 3
    n_per_class = 20
    n_features = 78  # 13 MFCCs × 6 statistics

    X_list, y_list = [], []
    for cls_idx in range(n_classes):
        samples = rng.normal(loc=float(cls_idx), scale=0.5,
                             size=(n_per_class, n_features)).astype(np.float32)
        X_list.append(samples)
        y_list.extend([cls_idx] * n_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    labels = ["a", "b", "c"]
    return X, y, labels
