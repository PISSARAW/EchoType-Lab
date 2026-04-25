"""Tests for echotype_lab.visualization.plots."""

from __future__ import annotations

import numpy as np
import pytest


class TestPlotWaveform:
    def test_returns_figure(self, sample_audio_int16):
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        from echotype_lab.visualization.plots import plot_waveform

        fig = plot_waveform(sample_audio_int16, show=False)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_save_to_file(self, tmp_path, sample_audio_int16):
        import matplotlib
        matplotlib.use("Agg")
        from echotype_lab.visualization.plots import plot_waveform

        out = tmp_path / "waveform.png"
        plot_waveform(sample_audio_int16, show=False, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotConfusionMatrix:
    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        from echotype_lab.visualization.plots import plot_confusion_matrix

        cm = np.array([[10, 2, 0], [1, 8, 3], [0, 1, 11]])
        fig = plot_confusion_matrix(cm, labels=["a", "b", "c"], show=False)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_normalized_matrix(self):
        import matplotlib
        matplotlib.use("Agg")
        from echotype_lab.visualization.plots import plot_confusion_matrix

        cm = np.eye(3, dtype=np.int64) * 5
        fig = plot_confusion_matrix(cm, normalize=True, show=False)
        assert fig is not None


class TestPlotClassDistribution:
    def test_basic(self):
        import matplotlib
        matplotlib.use("Agg")
        from echotype_lab.visualization.plots import plot_class_distribution

        y = np.array([0, 0, 1, 1, 1, 2])
        fig = plot_class_distribution(y, labels=["a", "b", "c"], show=False)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)
