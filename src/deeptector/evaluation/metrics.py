"""Robust binary classification metrics."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Compute metrics; undefined ranking metrics are represented by null."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and equal-length")
    predictions = [int(score >= threshold) for score in scores]
    has_both_classes = len(set(labels)) == 2
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    true_negative, false_positive = matrix[0]
    recall = float(recall_score(labels, predictions, zero_division=0))
    negative_count = true_negative + false_positive
    specificity = float(true_negative / negative_count) if negative_count else None
    balanced_accuracy = (
        float((recall + specificity) / 2) if has_both_classes and specificity is not None else None
    )
    roc_auc = float(roc_auc_score(labels, scores)) if has_both_classes else None
    pr_auc = float(average_precision_score(labels, scores)) if has_both_classes else None
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": recall,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": matrix,
        "threshold": threshold,
    }
