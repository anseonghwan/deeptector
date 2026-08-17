"""Traceable frame- and video-level evaluation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .aggregation import aggregate_scores
from .metrics import binary_metrics


class Evaluator:
    """Evaluate a detector without dataset-specific model logic."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        aggregation: str = "mean",
        top_k: int = 3,
        threshold: float = 0.5,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.aggregation = aggregation
        self.top_k = top_k
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader[dict[str, object]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return metrics and frame prediction rows."""
        self.model.eval()
        predictions: list[dict[str, Any]] = []
        for batch in loader:
            images = batch["image"].to(self.device)  # type: ignore[union-attr]
            output = self.model(images)
            logits = output.logits.detach().cpu().tolist()
            scores = output.score.detach().cpu().tolist()
            labels = batch["label"].int().tolist()  # type: ignore[union-attr]
            frame_indices = batch["frame_index"].tolist()  # type: ignore[union-attr]
            for index in range(len(labels)):
                predictions.append(
                    {
                        "dataset": batch["dataset"][index],
                        "video_id": batch["video_id"][index],
                        "frame_index": frame_indices[index],
                        "label": labels[index],
                        "logit": logits[index],
                        "score": scores[index],
                        "face_detected": bool(batch["face_detected"][index]),
                    }
                )
        if not predictions:
            raise ValueError("Evaluation loader yielded no samples")
        frame_metrics = binary_metrics(
            [row["label"] for row in predictions],
            [row["score"] for row in predictions],
            threshold=self.threshold,
        )
        by_video: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            by_video[(row["dataset"], row["video_id"])].append(row)
        video_labels, video_scores = [], []
        for key, rows in by_video.items():
            labels = {row["label"] for row in rows}
            if len(labels) != 1:
                raise ValueError(f"Inconsistent frame labels for video {key}")
            video_labels.append(labels.pop())
            video_scores.append(
                aggregate_scores([row["score"] for row in rows], self.aggregation, top_k=self.top_k)
            )
        metrics = {
            "frame_level": frame_metrics,
            "video_level": binary_metrics(video_labels, video_scores, threshold=self.threshold),
            "aggregation": {"strategy": self.aggregation, "top_k": self.top_k},
            "counts": {"frames": len(predictions), "videos": len(by_video)},
            "face_detection_failure_rate": (
                sum(not row["face_detected"] for row in predictions) / len(predictions)
            ),
        }
        return metrics, predictions


def save_evaluation(
    run_directory: str | Path,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    experiment: dict[str, Any] | None = None,
) -> None:
    """Save auditable JSON metrics and frame-level CSV predictions."""
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": experiment or {}, "metrics": metrics}
    (directory / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
