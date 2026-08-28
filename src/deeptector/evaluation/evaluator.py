"""Traceable frame- and video-level evaluation."""

from __future__ import annotations

import csv
import json
import math
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
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Return metrics plus auditable frame and video prediction rows."""
        self.model.eval()
        predictions: list[dict[str, Any]] = []
        for batch in loader:
            images = batch["image"].to(self.device)  # type: ignore[union-attr]
            output = self.model(images)
            if not torch.isfinite(output.logits).all() or not torch.isfinite(output.score).all():
                raise FloatingPointError("Evaluation produced NaN or Inf logits/scores")
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
                        "prediction": int(scores[index] >= self.threshold),
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
        video_predictions: list[dict[str, Any]] = []
        for key, rows in by_video.items():
            labels = {row["label"] for row in rows}
            if len(labels) != 1:
                raise ValueError(f"Inconsistent frame labels for video {key}")
            label = labels.pop()
            score = aggregate_scores(
                [row["score"] for row in rows], self.aggregation, top_k=self.top_k
            )
            if not math.isfinite(score):
                raise FloatingPointError(f"Video aggregation produced NaN or Inf for {key}")
            video_predictions.append(
                {
                    "dataset": key[0],
                    "video_id": key[1],
                    "label": label,
                    "score": score,
                    "prediction": int(score >= self.threshold),
                    "frame_count": len(rows),
                    "aggregation": self.aggregation,
                }
            )
        failed_faces = sum(not row["face_detected"] for row in predictions)
        metrics = {
            "frame_level": frame_metrics,
            "video_level": binary_metrics(
                [row["label"] for row in video_predictions],
                [row["score"] for row in video_predictions],
                threshold=self.threshold,
            ),
            "aggregation": {"strategy": self.aggregation, "top_k": self.top_k},
            "counts": {"frames": len(predictions), "videos": len(by_video)},
            "face_detection": {
                "attempted": len(predictions),
                "failed": failed_faces,
                "failure_rate": failed_faces / len(predictions),
            },
        }
        return metrics, predictions, video_predictions


def save_evaluation(
    run_directory: str | Path,
    metrics: dict[str, Any],
    frame_predictions: list[dict[str, Any]],
    video_predictions: list[dict[str, Any]],
    *,
    experiment: dict[str, Any] | None = None,
) -> None:
    """Save the stable M1 evaluation artifact contract."""
    directory = Path(run_directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Evaluation directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": experiment or {}, "metrics": metrics}
    (directory / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(directory / "frame_predictions.csv", frame_predictions)
    _write_csv(directory / "video_predictions.csv", video_predictions)
    contract = {
        "schema_version": 1,
        "artifact_type": "deeptector_m1_evaluation",
        "run_id": (experiment or {}).get("run_id"),
        "evaluated_at_utc": (experiment or {}).get("evaluated_at_utc"),
        "files": {
            "metrics": "metrics.json",
            "frame_predictions": "frame_predictions.csv",
            "video_predictions": "video_predictions.csv",
            "config_snapshot": "config.yaml",
        },
    }
    (directory / "artifact_manifest.json").write_text(
        json.dumps(contract, indent=2, allow_nan=False), encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty prediction artifact: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
