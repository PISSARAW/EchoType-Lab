"""
train_model.py
==============
Command-line script for training a keystroke classifier.

Usage
-----
    python -m scripts.train_model --data data/raw --model sklearn
    echotype-train --data data/raw --model sklearn --output data/models/clf.pkl

The script:
  1. Loads WAV files from the labelled data directory.
  2. Extracts MFCC features.
  3. Splits data into train / test sets.
  4. Trains the chosen classifier.
  5. Evaluates and prints a classification report.
  6. Saves the model to disk.
  7. Optionally saves a confusion-matrix plot.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="echotype-train",
        description="Train a keystroke classifier from labelled audio clips.",
    )
    parser.add_argument(
        "--data", default="data/raw", metavar="DIR",
        help="Root data directory produced by echotype-record (default: data/raw).",
    )
    parser.add_argument(
        "--model", choices=["sklearn", "tensorflow"], default="sklearn",
        help="Classifier backend (default: sklearn).",
    )
    parser.add_argument(
        "--output", default="data/models/classifier.pkl", metavar="PATH",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2, metavar="FRAC",
        help="Fraction of data held out for testing (default: 0.2).",
    )
    parser.add_argument(
        "--n-mfcc", type=int, default=13, metavar="N",
        help="Number of MFCC coefficients (default: 13).",
    )
    parser.add_argument(
        "--plot-cm", metavar="PATH",
        help="If provided, save a confusion-matrix plot to this path.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (TensorFlow backend only, default: 50).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        logger.error("Run echotype-record first to collect audio clips.")
        return 1

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    from echotype_lab.features.mfcc import extract_mfcc_batch  # noqa: PLC0415

    logger.info("Extracting MFCC features from %s …", data_dir)
    try:
        X, y, labels = extract_mfcc_batch(data_dir, n_mfcc=args.n_mfcc)
    except ValueError as exc:
        logger.error("Feature extraction failed: %s", exc)
        return 1

    print(f"Dataset: {len(X)} samples, {len(labels)} classes: {labels}")

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    if args.model == "sklearn":
        from echotype_lab.models.classifier import SklearnClassifier  # noqa: PLC0415
        clf = SklearnClassifier(labels=labels)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        save_path = Path(args.output)
        clf.save(save_path)
    else:  # tensorflow
        from echotype_lab.models.classifier import TFClassifier  # noqa: PLC0415
        clf = TFClassifier(
            n_features=X_train.shape[1],
            n_classes=len(labels),
            labels=labels,
        )
        clf.fit(X_train, y_train, epochs=args.epochs)
        y_pred = clf.predict(X_test)
        save_path = Path(args.output).with_suffix("")  # directory for SavedModel
        clf.save(save_path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    from echotype_lab.evaluation.metrics import print_report  # noqa: PLC0415

    print_report(y_test, y_pred, labels=labels)

    # ------------------------------------------------------------------
    # Optional confusion-matrix plot
    # ------------------------------------------------------------------
    if args.plot_cm:
        from echotype_lab.evaluation.metrics import compute_confusion_matrix  # noqa: PLC0415
        from echotype_lab.visualization.plots import plot_confusion_matrix  # noqa: PLC0415

        cm = compute_confusion_matrix(y_test, y_pred, n_classes=len(labels))
        plot_confusion_matrix(cm, labels=labels, save_path=args.plot_cm, normalize=True)
        print(f"Confusion matrix saved to {args.plot_cm}")

    print(f"\nModel saved to {save_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
