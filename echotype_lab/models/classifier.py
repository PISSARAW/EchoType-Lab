"""
echotype_lab.models.classifier
================================
Machine-learning classifiers for keystroke audio.

Two backends are provided:

* **SklearnClassifier** – wraps any scikit-learn estimator (default:
  ``RandomForestClassifier``).  Lightweight, no GPU required.
* **TFClassifier** – a small dense neural network built with TensorFlow /
  Keras.  Optional; only imported when TensorFlow is available.

Both classes share a common interface::

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    clf.save(path)
    clf2 = SklearnClassifier.load(path)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scikit-learn backend
# ---------------------------------------------------------------------------

class SklearnClassifier:
    """Keystroke classifier backed by a scikit-learn estimator.

    Parameters
    ----------
    estimator:
        Any scikit-learn compatible estimator.  Defaults to
        ``RandomForestClassifier(n_estimators=200, random_state=42)``.
    labels:
        Optional list of class names (used for display only).
    """

    def __init__(
        self,
        estimator: Any = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        if estimator is None:
            from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
            estimator = RandomForestClassifier(n_estimators=200, random_state=42)
        self.estimator = estimator
        self.labels = labels or []

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnClassifier":
        """Fit the estimator on training data.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.
        y:
            Integer label array of shape ``(n_samples,)``.

        Returns
        -------
        SklearnClassifier
            *self* for method chaining.
        """
        logger.info("Training sklearn classifier on %d samples …", len(X))
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for *X*.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            1-D int64 array of predicted class indices.
        """
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for *X*.

        Returns
        -------
        numpy.ndarray
            2-D float array of shape ``(n_samples, n_classes)``.
        """
        return self.estimator.predict_proba(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Serialize the classifier to *path* (pickle).

        Parameters
        ----------
        path:
            Destination ``.pkl`` file.

        Returns
        -------
        Path
            Resolved path of the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"estimator": self.estimator, "labels": self.labels}, fh)
        logger.info("Saved classifier to %s", path)
        return path.resolve()

    @classmethod
    def load(cls, path: Path | str) -> "SklearnClassifier":
        """Load a previously saved classifier from *path*.

        Parameters
        ----------
        path:
            Path to a ``.pkl`` file created by :meth:`save`.

        Returns
        -------
        SklearnClassifier
            Loaded instance.
        """
        path = Path(path)
        with path.open("rb") as fh:
            data = pickle.load(fh)  # noqa: S301 – trusted local file
        obj = cls.__new__(cls)
        obj.estimator = data["estimator"]
        obj.labels = data.get("labels", [])
        logger.info("Loaded classifier from %s", path)
        return obj


# ---------------------------------------------------------------------------
# TensorFlow / Keras backend (optional)
# ---------------------------------------------------------------------------

class TFClassifier:
    """Small dense neural-network classifier built with TensorFlow / Keras.

    Parameters
    ----------
    n_features:
        Dimensionality of the input feature vector.
    n_classes:
        Number of output classes.
    hidden_units:
        Sequence of hidden-layer sizes.
    dropout_rate:
        Dropout applied after each hidden layer (0 = disabled).
    labels:
        Optional list of class names.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_units: tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.3,
        labels: Optional[List[str]] = None,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.labels = labels or []
        self._model = self._build_model()

    def _build_model(self) -> Any:
        try:
            import tensorflow as tf  # noqa: PLC0415
            from tensorflow import keras  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for TFClassifier. "
                "Install it with: pip install tensorflow"
            ) from exc

        inputs = keras.Input(shape=(self.n_features,))
        x = inputs
        for units in self.hidden_units:
            x = keras.layers.Dense(units, activation="relu")(x)
            if self.dropout_rate > 0:
                x = keras.layers.Dropout(self.dropout_rate)(x)
        outputs = keras.layers.Dense(self.n_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: int = 1,
    ) -> "TFClassifier":
        """Train the neural network.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.
        y:
            Integer label array.
        epochs:
            Number of training epochs.
        batch_size:
            Mini-batch size.
        validation_split:
            Fraction of training data held out for validation.
        verbose:
            Keras verbosity level (0 = silent, 1 = progress bar, 2 = epoch).

        Returns
        -------
        TFClassifier
            *self* for method chaining.
        """
        logger.info("Training TFClassifier for %d epochs …", epochs)
        self._model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices.

        Returns
        -------
        numpy.ndarray
            1-D int64 array of length ``n_samples``.
        """
        proba = self._model.predict(X, verbose=0)
        return np.argmax(proba, axis=1).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Returns
        -------
        numpy.ndarray
            2-D float array of shape ``(n_samples, n_classes)``.
        """
        return self._model.predict(X, verbose=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Save the Keras model to *path* (SavedModel format).

        Parameters
        ----------
        path:
            Directory path for the SavedModel.

        Returns
        -------
        Path
            Resolved path.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        logger.info("Saved TFClassifier to %s", path)
        return path.resolve()

    @classmethod
    def load(cls, path: Path | str) -> "TFClassifier":
        """Load a previously saved TFClassifier.

        Parameters
        ----------
        path:
            Directory of a SavedModel created by :meth:`save`.

        Returns
        -------
        TFClassifier
            Loaded instance (model weights restored).
        """
        try:
            import tensorflow as tf  # noqa: PLC0415
            from tensorflow import keras  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required. Install with: pip install tensorflow"
            ) from exc

        path = Path(path)
        obj = cls.__new__(cls)
        obj._model = keras.models.load_model(str(path))
        obj.labels = []
        logger.info("Loaded TFClassifier from %s", path)
        return obj
