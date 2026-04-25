"""Tests for echotype_lab.audio.recorder."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


class TestAudioRecorder:
    """Tests for the AudioRecorder helper (no real microphone needed)."""

    def test_save_wav_creates_file(self, tmp_path, sample_audio_int16):
        from echotype_lab.audio.recorder import AudioRecorder

        rec = AudioRecorder()
        dest = tmp_path / "test.wav"
        rec.save_wav(sample_audio_int16, dest)

        assert dest.exists(), "WAV file was not created"

    def test_saved_wav_is_valid(self, tmp_path, sample_audio_int16):
        from echotype_lab.audio.recorder import AudioRecorder

        rec = AudioRecorder(sample_rate=44_100, channels=1)
        dest = tmp_path / "clip.wav"
        rec.save_wav(sample_audio_int16, dest)

        with wave.open(str(dest), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 44_100
            assert wf.getsampwidth() == 2  # 16-bit

    def test_save_wav_creates_parent_dirs(self, tmp_path, sample_audio_int16):
        from echotype_lab.audio.recorder import AudioRecorder

        rec = AudioRecorder()
        nested = tmp_path / "a" / "b" / "clip.wav"
        rec.save_wav(sample_audio_int16, nested)
        assert nested.exists()

    def test_save_wav_returns_resolved_path(self, tmp_path, sample_audio_int16):
        from echotype_lab.audio.recorder import AudioRecorder

        rec = AudioRecorder()
        dest = tmp_path / "clip.wav"
        result = rec.save_wav(sample_audio_int16, dest)
        assert result == dest.resolve()


class TestKeystrokeRecorder:
    """Tests for KeystrokeRecorder (filesystem logic only)."""

    def test_next_index_empty_dir(self, tmp_path):
        from echotype_lab.audio.recorder import KeystrokeRecorder

        rec = KeystrokeRecorder(output_dir=tmp_path, label="x")
        assert rec._index == 1

    def test_next_index_existing_files(self, tmp_path):
        from echotype_lab.audio.recorder import KeystrokeRecorder

        label_dir = tmp_path / "x"
        label_dir.mkdir()
        for i in range(3):
            (label_dir / f"x_{i:04d}.wav").touch()

        rec = KeystrokeRecorder(output_dir=tmp_path, label="x")
        assert rec._index == 4  # 3 existing + 1

    def test_build_path_structure(self, tmp_path):
        from echotype_lab.audio.recorder import KeystrokeRecorder

        rec = KeystrokeRecorder(output_dir=tmp_path, label="space")
        rec._index = 5
        path = rec._build_path()
        assert path.parent.name == "space"
        assert path.name == "space_0005.wav"
