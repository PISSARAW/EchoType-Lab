"""
echotype_lab.evaluation.metrics
================================
Evaluation helpers for keystroke classification.

Functions
---------
- :func:`compute_accuracy`
- :func:`compute_confusion_matrix`
- :func:`classification_report_dict`
- :func:`top_k_accuracy`
- :func:`print_report`
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of correctly classified samples.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    return float(np.mean(y_true == y_pred))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: Optional[int] = None,
) -> np.ndarray:
    """Compute a confusion matrix.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.
    n_classes:
        Total number of classes.  Inferred from the data if not provided.

    Returns
    -------
    numpy.ndarray
        2-D integer array of shape ``(n_classes, n_classes)`` where
        ``cm[i, j]`` is the number of samples truly of class *i* predicted
        as class *j*.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Return per-class precision, recall, F1, and support as a dict.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.
    labels:
        Human-readable class names.  If omitted, numeric strings are used.

    Returns
    -------
    dict
        Keys are class labels; values are dicts with keys
        ``precision``, ``recall``, ``f1``, ``support``.
        The special key ``"overall"`` holds macro-averaged scores.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if labels is None:
        labels = [str(c) for c in classes]

    report: Dict[str, Dict[str, float]] = {}
    precision_list, recall_list, f1_list = [], [], []

    for cls_idx, cls_name in zip(classes, labels):
        tp = int(((y_true == cls_idx) & (y_pred == cls_idx)).sum())
        fp = int(((y_true != cls_idx) & (y_pred == cls_idx)).sum())
        fn = int(((y_true == cls_idx) & (y_pred != cls_idx)).sum())
        support = int((y_true == cls_idx).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        report[cls_name] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   float(support),
        }
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    report["overall"] = {
        "precision": round(float(np.mean(precision_list)), 4),
        "recall":    round(float(np.mean(recall_list)), 4),
        "f1":        round(float(np.mean(f1_list)), 4),
        "support":   float(len(y_true)),
    }
    return report


def top_k_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int = 3,
) -> float:
    """Compute top-*k* accuracy (useful when predicting multiple candidates).

    Parameters
    ----------
    y_true:
        1-D array of ground-truth class indices.
    y_proba:
        2-D array of shape ``(n_samples, n_classes)`` with probability
        estimates.
    k:
        Number of top candidates to consider.

    Returns
    -------
    float
        Top-*k* accuracy in ``[0, 1]``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_proba = np.asarray(y_proba)

    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = sum(
        int(true in preds) for true, preds in zip(y_true, top_k_preds)
    )
    return correct / len(y_true)


def print_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> None:
    """Pretty-print a classification report to stdout.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.
    labels:
        Human-readable class names.
    """
    acc = compute_accuracy(y_true, y_pred)
    report = classification_report_dict(y_true, y_pred, labels=labels)

    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    for cls_name, scores in report.items():
        if cls_name == "overall":
            continue
        print(
            f"{cls_name:<15} "
            f"{scores['precision']:>10.4f} "
            f"{scores['recall']:>10.4f} "
            f"{scores['f1']:>10.4f} "
            f"{int(scores['support']):>10}"
        )
    print("-" * 60)
    ov = report["overall"]
    print(
        f"{'macro avg':<15} "
        f"{ov['precision']:>10.4f} "
        f"{ov['recall']:>10.4f} "
        f"{ov['f1']:>10.4f} "
        f"{int(ov['support']):>10}"
    )
    print(f"\nOverall accuracy: {acc:.4f}")
