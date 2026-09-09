"""Train the frozen visual foundation-model baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from deeptector.cli.common import (
    build_loader,
    build_model,
    load_config,
    save_config,
    validate_configured_data_protocol,
)
from deeptector.data.manifest import assert_no_split_leakage, load_manifest
from deeptector.training.trainer import Trainer, create_optimizer
from deeptector.utils.device import select_device
from deeptector.utils.logging import configure_logging
from deeptector.utils.seed import seed_everything


def validate_lomo_run_state(
    run_dir: str | Path,
    *,
    resume_from: str | Path | None,
    config: dict[str, object],
) -> None:
    """Fail closed when a LOMO invocation could overwrite or mix training state."""
    run_dir = Path(run_dir)
    best_checkpoint = run_dir / "checkpoint.pt"
    last_checkpoint = run_dir / "last_checkpoint.pt"
    config_snapshot = run_dir / "config.yaml"

    if config_snapshot.exists() and load_config(config_snapshot) != config:
        raise ValueError("Existing LOMO run config does not match the requested experiment config")

    if resume_from is not None:
        if Path(resume_from).resolve() != last_checkpoint.resolve():
            raise ValueError(
                "LOMO resume must use last_checkpoint.pt in the configured experiment run directory"
            )
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"LOMO resume checkpoint does not exist: {last_checkpoint}")
        if not best_checkpoint.is_file():
            raise FileNotFoundError(
                f"LOMO best checkpoint is missing beside resume state: {best_checkpoint}"
            )
        return

    training_state = [
        path
        for path in (best_checkpoint, last_checkpoint, *run_dir.glob("*.pt.tmp"))
        if path.exists()
    ]
    if training_state:
        names = ", ".join(sorted(path.name for path in training_state))
        raise FileExistsError(
            "Existing LOMO training state requires an explicit --resume-from "
            f"last_checkpoint.pt; found: {names}"
        )


def main() -> None:
    """Run manifest-driven training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/baseline.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume from a last_checkpoint.pt produced at a successful epoch boundary",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    protocol_binding = validate_configured_data_protocol(config["data"])
    run_dir = Path(args.run_dir or "runs") / config["experiment_name"]
    if protocol_binding is not None:
        try:
            validate_lomo_run_state(run_dir, resume_from=args.resume_from, config=config)
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            parser.error(str(error))
    elif args.resume_from and Path(args.resume_from).resolve().parent != run_dir.resolve():
        parser.error("--resume-from must point inside the configured experiment run directory")
    configure_logging(run_dir / "training.log")
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
    resume_state = (
        trainer.load_resume_state(args.resume_from, train_loader) if args.resume_from else None
    )
    save_config(config, run_dir / "config.yaml")
    trainer.fit(
        train_loader,
        validation_loader,
        int(train_config.get("epochs", 20)),
        resume_state=resume_state,
    )


if __name__ == "__main__":
    main()
