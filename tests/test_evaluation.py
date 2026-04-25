"""Tests for echotype_lab.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest


class TestComputeAccuracy:
    def test_perfect(self):
        from echotype_lab.evaluation.metrics import compute_accuracy

        y = np.array([0, 1, 2, 1, 0])
        assert compute_accuracy(y, y) == 1.0

    def test_zero(self):
        from echotype_lab.evaluation.metrics import compute_accuracy

        y_true = np.array([0, 1, 2])
        y_pred = np.array([2, 0, 1])
        assert compute_accuracy(y_true, y_pred) == 0.0

    def test_partial(self):
        from echotype_lab.evaluation.metrics import compute_accuracy

        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 0, 0])  # 3/4 correct
        assert abs(compute_accuracy(y_true, y_pred) - 0.75) < 1e-9

    def test_empty_raises(self):
        from echotype_lab.evaluation.metrics import compute_accuracy

        with pytest.raises(ValueError):
            compute_accuracy(np.array([]), np.array([]))


class TestComputeConfusionMatrix:
    def test_shape(self):
        from echotype_lab.evaluation.metrics import compute_confusion_matrix

        y = np.array([0, 1, 2, 0, 1, 2])
        cm = compute_confusion_matrix(y, y, n_classes=3)
        assert cm.shape == (3, 3)

    def test_diagonal_for_perfect_predictions(self):
        from echotype_lab.evaluation.metrics import compute_confusion_matrix

        y = np.array([0, 1, 2, 0, 1, 2])
        cm = compute_confusion_matrix(y, y, n_classes=3)
        assert np.all(cm == np.diag(np.diag(cm)))

    def test_counts_correct(self):
        from echotype_lab.evaluation.metrics import compute_confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = compute_confusion_matrix(y_true, y_pred, n_classes=2)
        # cm[i, j] = number of true-i predicted-as-j
        assert cm[0, 0] == 1  # true 0, pred 0
        assert cm[0, 1] == 1  # true 0, pred 1
        assert cm[1, 0] == 1  # true 1, pred 0
        assert cm[1, 1] == 1  # true 1, pred 1


class TestClassificationReportDict:
    def test_overall_key_present(self):
        from echotype_lab.evaluation.metrics import classification_report_dict

        y = np.array([0, 1, 2, 0, 1, 2])
        report = classification_report_dict(y, y, labels=["a", "b", "c"])
        assert "overall" in report

    def test_perfect_precision_recall_f1(self):
        from echotype_lab.evaluation.metrics import classification_report_dict

        y = np.array([0, 1, 2])
        report = classification_report_dict(y, y, labels=["a", "b", "c"])
        for cls in ["a", "b", "c"]:
            assert report[cls]["precision"] == 1.0
            assert report[cls]["recall"] == 1.0
            assert report[cls]["f1"] == 1.0

    def test_support_counts(self):
        from echotype_lab.evaluation.metrics import classification_report_dict

        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        report = classification_report_dict(y_true, y_pred, labels=["a", "b"])
        assert report["a"]["support"] == 2.0
        assert report["b"]["support"] == 3.0


class TestTopKAccuracy:
    def test_top1_equals_accuracy(self):
        from echotype_lab.evaluation.metrics import top_k_accuracy, compute_accuracy

        y_true = np.array([0, 1, 2, 0])
        y_proba = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
        ])
        acc_k1 = top_k_accuracy(y_true, y_proba, k=1)
        acc = compute_accuracy(y_true, np.argmax(y_proba, axis=1))
        assert abs(acc_k1 - acc) < 1e-9

    def test_top_k_geq_top_1(self):
        from echotype_lab.evaluation.metrics import top_k_accuracy

        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 5, size=50)
        y_proba = rng.dirichlet(np.ones(5), size=50)
        acc1 = top_k_accuracy(y_true, y_proba, k=1)
        acc3 = top_k_accuracy(y_true, y_proba, k=3)
        assert acc3 >= acc1
