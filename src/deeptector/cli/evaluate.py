"""Evaluate in-dataset or cross-dataset manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from deeptector.cli.common import build_loader, build_model, load_config, save_config
from deeptector.data.manifest import load_manifest
from deeptector.evaluation.evaluator import Evaluator, save_evaluation
from deeptector.training.checkpoint import load_checkpoint
from deeptector.utils.device import select_device
from deeptector.utils.seed import seed_everything


def main() -> None:
    """Run traceable frame- and video-level evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Override test manifest for cross-dataset evaluation",
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--evaluation-id", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = select_device(args.device)
    model = build_model(config["model"])
    metadata = load_checkpoint(args.checkpoint, model, map_location=device)
    data_config, eval_config = config["data"], config["evaluation"]
    manifest_key = "validation_manifest" if args.split == "validation" else "test_manifest"
    manifest = args.manifest or data_config[manifest_key]
    loader = build_loader(
        manifest,
        data_config,
        split=args.split,
        batch_size=int(config["train"].get("batch_size", 32)),
        num_workers=int(config["train"].get("num_workers", 0)),
    )
    evaluator = Evaluator(
        model,
        device,
        aggregation=eval_config.get("aggregation", "mean"),
        top_k=int(eval_config.get("top_k", 3)),
        threshold=float(eval_config.get("threshold", 0.5)),
    )
    metrics, frame_predictions, video_predictions = evaluator.evaluate(loader)
    evaluated_at = datetime.now(timezone.utc)
    run_id = args.evaluation_id or evaluated_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (
        Path(args.run_dir or "runs")
        / config["experiment_name"]
        / "evaluations"
        / args.split
        / run_id
    )
    datasets = sorted({row["dataset"] for row in frame_predictions})
    training_datasets = sorted(
        {
            record.dataset
            for record in load_manifest(data_config["train_manifest"])
            if record.split == "train"
        }
    )
    validation_datasets = sorted(
        {
            record.dataset
            for record in load_manifest(data_config["validation_manifest"])
            if record.split == "validation"
        }
    )
    experiment = {
        "training_dataset": training_datasets,
        "validation_dataset": validation_datasets,
        "testing_dataset": datasets,
        "evaluation_type": "cross-dataset" if args.manifest else "in-dataset",
        "evaluated_split": args.split,
        "run_id": run_id,
        "evaluated_at_utc": evaluated_at.isoformat(),
        "model_name": config["model"]["name"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": metadata.get("epoch"),
        "aggregation": eval_config.get("aggregation", "mean"),
        "threshold": float(eval_config.get("threshold", 0.5)),
        "config_snapshot": "config.yaml",
        "evaluated_label_counts": dict(
            sorted(Counter(row["label"] for row in video_predictions).items())
        ),
        "sampling_configuration": data_config,
        "preprocessing_configuration": {
            key: data_config.get(key)
            for key in ("input_resolution", "face_margin", "normalization")
        },
    }
    save_evaluation(
        run_dir,
        metrics,
        frame_predictions,
        video_predictions,
        experiment=experiment,
    )
    save_config(config, run_dir / "config.yaml")
    print(f"Evaluation artifacts: {run_dir}")


if __name__ == "__main__":
    main()
