import pytest
import torch
from helpers import TinyEncoder
from torch.utils.data import DataLoader, Dataset

from deeptector.cli.evaluate import select_evaluation_manifest
from deeptector.evaluation.aggregation import aggregate_scores
from deeptector.evaluation.evaluator import Evaluator, save_evaluation
from deeptector.evaluation.metrics import binary_metrics
from deeptector.models.classifier import DeepfakeClassifier


def test_video_aggregation_strategies():
    scores = [0.1, 0.4, 0.9]
    assert aggregate_scores(scores, "mean") == pytest.approx(0.4666667)
    assert aggregate_scores(scores, "max") == 0.9
    assert aggregate_scores(scores, "top_k_mean", top_k=2) == pytest.approx(0.65)


def test_binary_metrics():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_binary_metrics_reports_specificity_and_balanced_accuracy():
    metrics = binary_metrics([0, 0, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["balanced_accuracy"] == pytest.approx(5 / 6)
    assert metrics["confusion_matrix"] == [[2, 1], [0, 1]]


class EvaluationDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        label = index // 2
        return {
            "image": torch.full((3, 8, 8), float(label)),
            "label": torch.tensor(float(label)),
            "dataset": "fixture",
            "video_id": f"video-{label}",
            "frame_index": index % 2,
            "face_detected": index != 0,
        }


def test_evaluation_persists_frame_and_video_contract(tmp_path):
    torch.manual_seed(4)
    evaluator = Evaluator(DeepfakeClassifier(TinyEncoder()), torch.device("cpu"))
    metrics, frames, videos = evaluator.evaluate(DataLoader(EvaluationDataset(), batch_size=2))
    experiment = {"run_id": "fixed", "evaluated_at_utc": "2026-08-25T00:00:00+00:00"}
    save_evaluation(tmp_path / "evaluation", metrics, frames, videos, experiment=experiment)
    names = {path.name for path in (tmp_path / "evaluation").iterdir()}
    assert names == {
        "artifact_manifest.json",
        "frame_predictions.csv",
        "metrics.json",
        "video_predictions.csv",
    }
    assert len(videos) == 2
    assert metrics["face_detection"] == {"attempted": 4, "failed": 1, "failure_rate": 0.25}
    with pytest.raises(FileExistsError):
        save_evaluation(tmp_path / "evaluation", metrics, frames, videos, experiment=experiment)


def test_protocol_bound_evaluation_rejects_manifest_override(tmp_path):
    configured = tmp_path / "deepfakes.csv"
    other = tmp_path / "faceswap.csv"
    data = {"validation_manifest": str(configured), "test_manifest": str(configured)}

    with pytest.raises(ValueError, match="cannot override"):
        select_evaluation_manifest(
            data,
            split="test",
            override=str(other),
            protocol_binding={"fold_slug": "deepfakes"},
        )

    assert select_evaluation_manifest(
        data,
        split="test",
        override=str(configured.resolve()),
        protocol_binding={"fold_slug": "deepfakes"},
    ) == str(configured.resolve())
