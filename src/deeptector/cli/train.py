"""Train the frozen visual foundation-model baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from deeptector.cli.common import build_loader, build_model, load_config, save_config
from deeptector.data.manifest import assert_no_split_leakage, load_manifest
from deeptector.training.trainer import Trainer, create_optimizer
from deeptector.utils.device import select_device
from deeptector.utils.logging import configure_logging
from deeptector.utils.seed import seed_everything


def main() -> None:
    """Run manifest-driven training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/baseline.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(args.run_dir or "runs") / config["experiment_name"]
    configure_logging(run_dir / "training.log")
    save_config(config, run_dir / "config.yaml")
    seed_everything(int(config.get("seed", 42)))
    device = select_device(args.device)
    train_config, data_config = config["train"], config["data"]
    combined_records = []
    for manifest_path in {
        data_config["train_manifest"],
        data_config["validation_manifest"],
        data_config.get("test_manifest"),
    }:
        if manifest_path:
            combined_records.extend(load_manifest(manifest_path))
    assert_no_split_leakage(
        combined_records,
        identity_disjoint=bool(data_config.get("identity_disjoint", False)),
    )
    model = build_model(config["model"])
    optimizer = create_optimizer(
        model,
        name=train_config.get("optimizer", "adamw"),
        learning_rate=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    common = {
        "batch_size": int(train_config.get("batch_size", 32)),
        "num_workers": int(train_config.get("num_workers", 0)),
    }
    train_loader = build_loader(
        data_config["train_manifest"],
        data_config,
        split="train",
        shuffle=True,
        balanced_sampling=bool(train_config.get("balanced_sampling", False)),
        seed=int(config.get("seed", 42)),
        **common,
    )
    validation_loader = build_loader(
        data_config["validation_manifest"], data_config, split="validation", **common
    )
    trainer = Trainer(
        model,
        optimizer,
        device,
        checkpoint_path=run_dir / "checkpoint.pt",
        patience=train_config.get("early_stopping_patience", 5),
        config=config,
    )
    trainer.fit(train_loader, validation_loader, int(train_config.get("epochs", 20)))


if __name__ == "__main__":
    main()
