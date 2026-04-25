"""Tests for echotype_lab.models.classifier (sklearn backend)."""

from __future__ import annotations

import numpy as np
import pytest


class TestSklearnClassifier:
    """Tests for SklearnClassifier."""

    def test_fit_and_predict(self, simple_dataset):
        from echotype_lab.models.classifier import SklearnClassifier

        X, y, labels = simple_dataset
        clf = SklearnClassifier(labels=labels)
        clf.fit(X, y)
        y_pred = clf.predict(X)

        assert y_pred.shape == y.shape
        assert set(y_pred.tolist()).issubset(set(y.tolist()))

    def test_fit_returns_self(self, simple_dataset):
        from echotype_lab.models.classifier import SklearnClassifier

        X, y, labels = simple_dataset
        clf = SklearnClassifier(labels=labels)
        result = clf.fit(X, y)
        assert result is clf

    def test_predict_proba_shape(self, simple_dataset):
        from echotype_lab.models.classifier import SklearnClassifier

        X, y, labels = simple_dataset
        clf = SklearnClassifier(labels=labels)
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        assert proba.shape == (len(X), len(labels))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_and_load(self, tmp_path, simple_dataset):
        from echotype_lab.models.classifier import SklearnClassifier

        X, y, labels = simple_dataset
        clf = SklearnClassifier(labels=labels)
        clf.fit(X, y)

        save_path = tmp_path / "clf.pkl"
        clf.save(save_path)
        assert save_path.exists()

        clf2 = SklearnClassifier.load(save_path)
        y_pred_orig = clf.predict(X)
        y_pred_load = clf2.predict(X)
        np.testing.assert_array_equal(y_pred_orig, y_pred_load)
        assert clf2.labels == labels

    def test_high_accuracy_on_separable_data(self, simple_dataset):
        from echotype_lab.models.classifier import SklearnClassifier
        from echotype_lab.evaluation.metrics import compute_accuracy

        X, y, labels = simple_dataset
        clf = SklearnClassifier(labels=labels)
        clf.fit(X, y)
        acc = compute_accuracy(y, clf.predict(X))
        # The synthetic data is well-separated; expect good accuracy
        assert acc > 0.8
