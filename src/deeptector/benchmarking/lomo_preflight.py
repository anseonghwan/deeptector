"""Bounded real-data preflight for configured FF++ LOMO folds."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from deeptector.cli.common import (
    build_model,
    load_config,
    validate_configured_data_protocol,
)
from deeptector.data.dataset import VideoFrameDataset
from deeptector.data.face_detection import OpenCVHaarFaceDetector
from deeptector.data.ffpp import MANIPULATIONS
from deeptector.data.manifest import VideoRecord, load_manifest
from deeptector.data.sampling import UniformFrameSampler
from deeptector.data.transforms import ImageTransform
from deeptector.training.trainer import Trainer, create_optimizer
from deeptector.utils.device import select_device
from deeptector.utils.seed import seed_everything


def select_lomo_preflight_records(
    records: list[VideoRecord], held_out_manipulation: str
) -> dict[str, list[VideoRecord]]:
    """Select one deterministic video per required split/manipulation category."""
    if held_out_manipulation not in MANIPULATIONS:
        raise ValueError(f"Unsupported held-out manipulation: {held_out_manipulation}")
    selected: dict[str, list[VideoRecord]] = {}
    for split in ("train", "validation", "test"):
        categories = (
            ("original", held_out_manipulation)
            if split == "test"
            else ("original", *(value for value in MANIPULATIONS if value != held_out_manipulation))
        )
        split_records = []
        for category in categories:
            matches = sorted(
                (
                    record
                    for record in records
                    if record.split == split and record.manipulation_type == category
                ),
                key=lambda record: record.video_id,
            )
            if not matches:
                raise ValueError(f"LOMO preflight has no {category} record in {split}")
            split_records.append(matches[0])
        selected[split] = split_records
    return selected


def run_lomo_preflight(
    config_path: str | Path,
    output_path: str | Path,
    *,
    device_request: str = "xpu",
) -> dict[str, Any]:
    """Run a bounded end-to-end finite-value check without saving checkpoints."""
    config_path = Path(config_path)
    config = load_config(config_path)
    binding = validate_configured_data_protocol(config["data"])
    if binding is None:
        raise ValueError("LOMO preflight requires a protocol-bound configuration")
    output = _ensure_output_under_runs(output_path)
    if output.exists():
        raise FileExistsError(f"LOMO preflight output already exists: {output}")
    started = time.perf_counter()
    report = {
        "schema_version": 2,
        "artifact_type": "deeptector_m2a_lomo_preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": config_path.as_posix(),
            "sha256": _sha256(config_path),
            "experiment_name": config["experiment_name"],
            "seed": int(config.get("seed", 42)),
        },
        "protocol_binding": binding,
        "device": {
            "requested": device_request,
            "selected": None,
            "name": None,
            "amp_enabled": False,
            "peak_memory_reset_error": None,
            "peak_allocated_bytes": None,
            "peak_allocated_error": None,
            "peak_reserved_bytes": None,
            "peak_reserved_error": None,
            "survived": False,
            "survival_error": None,
        },
        "phases": {},
        "training": {"optimizer_steps": 0, "amp_overflow_skips": 0},
        "finite_model_parameters": False,
        "elapsed_seconds": 0.0,
        "success": False,
        "checkpoint_written": False,
    }
    try:
        seed_everything(int(config.get("seed", 42)))
        device = select_device(device_request)
        report["device"].update({"selected": str(device), "name": _device_name(device)})
        if device.type != "xpu":
            raise ValueError("LOMO hardware preflight requires an Intel XPU device")

        data_config = config["data"]
        records = load_manifest(data_config["train_manifest"])
        selected = select_lomo_preflight_records(records, binding["held_out_manipulation"])
        model = build_model(config["model"])
        optimizer = create_optimizer(
            model,
            name=config["train"].get("optimizer", "adamw"),
            learning_rate=float(config["train"].get("learning_rate", 1e-3)),
            weight_decay=float(config["train"].get("weight_decay", 1e-4)),
        )
        trainer = Trainer(
            model,
            optimizer,
            device,
            checkpoint_path=output.parent / "preflight-unused-checkpoint.pt",
            config=config,
        )
        report["device"]["amp_enabled"] = trainer.amp_enabled
        torch.xpu.empty_cache()
        report["device"]["peak_memory_reset_error"] = _reset_xpu_peak_memory(device)

        for split in ("train", "validation", "test"):
            loader = _build_preflight_loader(
                selected[split],
                data_config,
                batch_size=int(config["train"].get("batch_size", 8)),
            )
            total_loss = 0.0
            frames = 0
            batches = 0
            phase_started = time.perf_counter()
            for batch in loader:
                if split == "test":
                    batch_size = _run_label_free_batch(
                        model,
                        batch,
                        device=device,
                        amp_enabled=trainer.amp_enabled,
                    )
                else:
                    loss, batch_size = trainer.run_batch(batch, training=split == "train")
                    total_loss += loss * batch_size
                frames += batch_size
                batches += 1
            _synchronize(device)
            phase_report = {
                "videos": len(selected[split]),
                "frames": frames,
                "batches": batches,
                "label_free": split == "test",
                "elapsed_seconds": time.perf_counter() - phase_started,
                "face_detection": {
                    "attempted": loader.dataset.face_stats.attempted,
                    "failed": loader.dataset.face_stats.failed,
                    "failure_rate": loader.dataset.face_stats.failure_rate,
                },
                "manipulation_counts": _manipulation_counts(selected[split]),
            }
            if split != "test":
                phase_report["mean_loss"] = total_loss / frames
            report["phases"][split] = phase_report

        finite_parameters = all(
            torch.isfinite(parameter).all().item() for parameter in model.parameters()
        )
        survived, survival_error = _device_survived(device)
        peak_allocated, allocated_error = _xpu_peak("max_memory_allocated", device)
        peak_reserved, reserved_error = _xpu_peak("max_memory_reserved", device)
        report["device"].update(
            {
                "peak_allocated_bytes": peak_allocated,
                "peak_allocated_error": allocated_error,
                "peak_reserved_bytes": peak_reserved,
                "peak_reserved_error": reserved_error,
                "survived": survived,
                "survival_error": survival_error,
            }
        )
        report["training"] = {
            "optimizer_steps": trainer.optimizer_steps,
            "amp_overflow_skips": trainer.amp_overflow_skips,
        }
        report["finite_model_parameters"] = finite_parameters
        report["success"] = all(
            (
                trainer.amp_enabled,
                trainer.optimizer_steps > 0,
                finite_parameters,
                survived,
                report["device"]["peak_memory_reset_error"] is None,
                peak_allocated is not None,
                peak_reserved is not None,
            )
        )
        if not report["success"]:
            raise RuntimeError("LOMO preflight health requirements were not satisfied")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_report(output, report)
        raise

    report["elapsed_seconds"] = time.perf_counter() - started
    _write_report(output, report)
    return report


def _run_label_free_batch(
    model: torch.nn.Module,
    batch: dict[str, object],
    *,
    device: torch.device,
    amp_enabled: bool,
) -> int:
    """Check the held-out forward path without consuming labels or producing metrics."""
    model.eval()
    images = batch["image"].to(device)  # type: ignore[union-attr]
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ),
    ):
        output = model(images)
    if not torch.isfinite(output.logits).all():
        raise FloatingPointError("Non-finite logits detected during label-free test preflight")
    return int(images.shape[0])


def _build_preflight_loader(
    records: list[VideoRecord], data_config: dict[str, Any], *, batch_size: int
) -> DataLoader[dict[str, object]]:
    dataset = VideoFrameDataset(
        records,
        UniformFrameSampler(int(data_config.get("frames_per_video", 8))),
        ImageTransform(
            int(data_config.get("input_resolution", 224)),
            str(data_config.get("normalization", "clip")),
        ),
        OpenCVHaarFaceDetector(),
        face_margin=float(data_config.get("face_margin", 0.2)),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )


def _manipulation_counts(records: list[VideoRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.manipulation_type)] += 1
    return dict(sorted(counts.items()))


def _synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_survived(device: torch.device) -> tuple[bool, str | None]:
    try:
        probe = torch.ones(8, device=device)
        valid = bool(torch.equal(probe + 1, torch.full_like(probe, 2)))
        _synchronize(device)
        return (True, None) if valid else (False, "Unexpected device survival result")
    except RuntimeError as error:
        return False, f"{type(error).__name__}: {error}"


def _device_name(device: torch.device) -> str:
    if device.type == "xpu":
        return torch.xpu.get_device_name(device.index or 0)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "CPU"


def _reset_xpu_peak_memory(device: torch.device) -> str | None:
    reset = getattr(torch.xpu, "reset_peak_memory_stats", None)
    if reset is None:
        return "torch.xpu.reset_peak_memory_stats is unavailable"
    try:
        reset(device.index or 0)
    except (RuntimeError, TypeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def _xpu_peak(name: str, device: torch.device) -> tuple[int | None, str | None]:
    if device.type != "xpu":
        return None, "Selected device is not XPU"
    function = getattr(torch.xpu, name, None)
    if function is None:
        return None, f"torch.xpu.{name} is unavailable"
    try:
        return int(function(device.index or 0)), None
    except (RuntimeError, TypeError) as error:
        return None, f"{type(error).__name__}: {error}"


def _ensure_output_under_runs(path: str | Path) -> Path:
    output = Path(path).resolve()
    runs = (Path.cwd() / "runs").resolve()
    if output != runs and runs not in output.parents:
        raise ValueError(f"LOMO preflight output must be written under {runs}: {output}")
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
