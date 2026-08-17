import pytest

from deeptector.evaluation.aggregation import aggregate_scores
from deeptector.evaluation.metrics import binary_metrics


def test_video_aggregation_strategies():
    scores = [0.1, 0.4, 0.9]
    assert aggregate_scores(scores, "mean") == pytest.approx(0.4666667)
    assert aggregate_scores(scores, "max") == 0.9
    assert aggregate_scores(scores, "top_k_mean", top_k=2) == pytest.approx(0.65)


def test_binary_metrics():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
