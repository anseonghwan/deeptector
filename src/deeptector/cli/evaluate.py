"""Evaluate in-dataset or cross-dataset manifests."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = select_device(args.device)
    model = build_model(config["model"])
    metadata = load_checkpoint(args.checkpoint, model, map_location=device)
    data_config, eval_config = config["data"], config["evaluation"]
    manifest = args.manifest or data_config["test_manifest"]
    loader = build_loader(
        manifest,
        data_config,
        split="test",
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
    metrics, predictions = evaluator.evaluate(loader)
    run_dir = Path(args.run_dir or "runs") / (config["experiment_name"] + "_evaluation")
    save_config(config, run_dir / "config.yaml")
    datasets = sorted({row["dataset"] for row in predictions})
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
        "model_name": config["model"]["name"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": metadata.get("epoch"),
        "sampling_configuration": data_config,
        "preprocessing_configuration": {
            key: data_config.get(key)
            for key in ("input_resolution", "face_margin", "normalization")
        },
    }
    save_evaluation(run_dir, metrics, predictions, experiment=experiment)


if __name__ == "__main__":
    main()
