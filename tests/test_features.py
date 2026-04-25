"""Tests for echotype_lab.features.mfcc."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 44_100) -> None:
    """Write int16 samples to a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


class TestExtractMfccFromArray:
    """Test feature extraction from raw arrays."""

    def test_output_shape_default(self, sample_audio_float32):
        from echotype_lab.features.mfcc import extract_mfcc_from_array

        feats = extract_mfcc_from_array(sample_audio_float32)
        assert feats.shape == (13 * 6,), f"Expected (78,), got {feats.shape}"

    def test_output_shape_custom_n_mfcc(self, sample_audio_float32):
        from echotype_lab.features.mfcc import extract_mfcc_from_array

        feats = extract_mfcc_from_array(sample_audio_float32, n_mfcc=20)
        assert feats.shape == (20 * 6,)

    def test_output_dtype(self, sample_audio_float32):
        from echotype_lab.features.mfcc import extract_mfcc_from_array

        feats = extract_mfcc_from_array(sample_audio_float32)
        assert feats.dtype == np.float32

    def test_int16_input_accepted(self, sample_audio_int16):
        from echotype_lab.features.mfcc import extract_mfcc_from_array

        feats = extract_mfcc_from_array(sample_audio_int16)
        assert feats.shape == (13 * 6,)
        assert np.all(np.isfinite(feats))

    def test_no_nan_or_inf(self, sample_audio_float32):
        from echotype_lab.features.mfcc import extract_mfcc_from_array

        feats = extract_mfcc_from_array(sample_audio_float32)
        assert np.all(np.isfinite(feats))


class TestExtractMfccFromFile:
    """Test feature extraction from WAV files."""

    def test_from_wav_file(self, tmp_path, sample_audio_int16):
        from echotype_lab.features.mfcc import extract_mfcc

        wav = tmp_path / "clip.wav"
        _write_wav(wav, sample_audio_int16)

        feats = extract_mfcc(wav)
        assert feats.shape == (13 * 6,)
        assert np.all(np.isfinite(feats))

    def test_missing_file_raises(self):
        from echotype_lab.features.mfcc import extract_mfcc

        with pytest.raises(Exception):
            extract_mfcc("does_not_exist.wav")


class TestExtractMfccBatch:
    """Test batch processing of a labelled dataset directory."""

    def _make_dataset(self, base: Path, labels, n_per_label=3,
                      sample_rate=44_100):
        rng = np.random.default_rng(0)
        for label in labels:
            for i in range(1, n_per_label + 1):
                samples = rng.integers(-32768, 32767, size=22_050,
                                       dtype=np.int16)
                _write_wav(base / label / f"{label}_{i:04d}.wav",
                           samples, sample_rate)

    def test_returns_correct_shapes(self, tmp_path):
        from echotype_lab.features.mfcc import extract_mfcc_batch

        labels = ["a", "b", "c"]
        n = 4
        self._make_dataset(tmp_path, labels, n_per_label=n)

        X, y, returned_labels = extract_mfcc_batch(tmp_path)
        total = len(labels) * n
        assert X.shape == (total, 13 * 6)
        assert y.shape == (total,)
        assert returned_labels == labels

    def test_empty_directory_raises(self, tmp_path):
        from echotype_lab.features.mfcc import extract_mfcc_batch

        with pytest.raises(ValueError):
            extract_mfcc_batch(tmp_path)
