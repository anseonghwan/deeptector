"""Bounded Intel XPU capacity measurement for the frozen M1 baseline."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from deeptector.cli.common import build_loader, build_model, load_config
from deeptector.training.trainer import Trainer, create_optimizer
from deeptector.utils.device import select_device
from deeptector.utils.seed import seed_everything

BATCH_SIZES = (2, 4, 8)
WARMUP_FRAMES = 8
MEASURED_FRAMES = 32
REPETITIONS = 3
GIB = 1024**3


class FixedOrderSampler(Sampler[int]):
    """Replay a materialized class-balanced sample sequence exactly."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _fixed_loader(
    dataset: Dataset[dict[str, object]], indices: list[int], batch_size: int
) -> DataLoader[dict[str, object]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=FixedOrderSampler(indices),
        num_workers=0,
        pin_memory=False,
    )


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _indices_digest(indices: list[int]) -> str:
    return hashlib.sha256(json.dumps(indices).encode("utf-8")).hexdigest()


def _host_memory_snapshot() -> dict[str, int | None]:
    if os.name != "nt":
        return {
            "total_physical_bytes": None,
            "available_physical_bytes": None,
            "process_rss_bytes": None,
            "process_private_bytes": None,
        }

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.DWORD),
            ("memory_load", wintypes.DWORD),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
            ("private_usage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatusEx)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    status = MemoryStatusEx()
    status.length = ctypes.sizeof(status)
    total = available = None
    if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        total = int(status.total_physical)
        available = int(status.available_physical)

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    handle = kernel32.GetCurrentProcess()
    rss = private = None
    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        rss = int(counters.working_set_size)
        private = int(counters.private_usage)
    return {
        "total_physical_bytes": total,
        "available_physical_bytes": available,
        "process_rss_bytes": rss,
        "process_private_bytes": private,
    }


def _xpu_call(name: str, device_index: int) -> Any:
    function = getattr(torch.xpu, name, None)
    if function is None:
        return None
    try:
        return function(device_index)
    except (RuntimeError, TypeError):
        return None


def _xpu_memory_snapshot(device_index: int) -> dict[str, Any]:
    mem_info = _xpu_call("mem_get_info", device_index)
    return {
        "allocated_bytes": _xpu_call("memory_allocated", device_index),
        "reserved_bytes": _xpu_call("memory_reserved", device_index),
        "max_allocated_bytes": _xpu_call("max_memory_allocated", device_index),
        "max_reserved_bytes": _xpu_call("max_memory_reserved", device_index),
        "mem_get_info_free_bytes": int(mem_info[0]) if mem_info is not None else None,
        "mem_get_info_total_bytes": int(mem_info[1]) if mem_info is not None else None,
    }


def _reset_xpu_peaks(device_index: int) -> None:
    empty_cache = getattr(torch.xpu, "empty_cache", None)
    if empty_cache is not None:
        empty_cache()
    reset = getattr(torch.xpu, "reset_peak_memory_stats", None)
    if reset is not None:
        reset(device_index)


def _device_survived(device: torch.device) -> tuple[bool, str | None]:
    try:
        probe = torch.ones(8, device=device)
        if not torch.equal(probe + 1, torch.full_like(probe, 2)):
            return False, "XPU survival tensor returned an unexpected value"
        torch.xpu.synchronize()
        return True, None
    except RuntimeError as error:
        return False, f"{type(error).__name__}: {error}"


def _failure_kind(error: Exception) -> str:
    message = str(error).lower()
    if isinstance(error, torch.OutOfMemoryError) or "out of memory" in message:
        return "xpu_oom"
    if "alloc" in message:
        return "allocation_failure"
    if "device" in message or "driver" in message or "level zero" in message:
        return "device_or_driver_error"
    if isinstance(error, FloatingPointError):
        return "non_finite"
    return "runtime_error"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def summarize_repetitions(
    repetitions: list[dict[str, Any]], *, expected_repetitions: int = REPETITIONS
) -> dict[str, Any]:
    successful = [row for row in repetitions if row.get("success")]
    training_steps = [step for row in successful for step in row.get("steps", [])]
    e2e_durations = [float(step["end_to_end_seconds"]) for step in training_steps]
    compute_durations = [float(step["compute_seconds"]) for step in training_steps]
    total_frames = sum(int(row.get("processed_frames", 0)) for row in successful)
    total_e2e = sum(e2e_durations)
    total_compute = sum(compute_durations)
    max_allocated = [
        row["xpu_memory_after"]["max_allocated_bytes"]
        for row in successful
        if row["xpu_memory_after"].get("max_allocated_bytes") is not None
    ]
    max_reserved = [
        row["xpu_memory_after"]["max_reserved_bytes"]
        for row in successful
        if row["xpu_memory_after"].get("max_reserved_bytes") is not None
    ]
    available = [
        value
        for row in successful
        for value in row.get("available_system_memory_samples_bytes", [])
        if value is not None
    ]
    process_rss = [
        value
        for row in successful
        for value in row.get("process_rss_samples_bytes", [])
        if value is not None
    ]
    stable = len(successful) == expected_repetitions and all(
        row.get("device_survived") for row in successful
    )
    return {
        "stable_repetitions": len(successful),
        "expected_repetitions": expected_repetitions,
        "stable": stable,
        "total_frames": total_frames,
        "total_end_to_end_seconds": total_e2e,
        "total_compute_seconds": total_compute,
        "end_to_end_frames_per_second": total_frames / total_e2e if total_e2e else None,
        "compute_frames_per_second": total_frames / total_compute if total_compute else None,
        "mean_end_to_end_step_seconds": statistics.fmean(e2e_durations) if e2e_durations else None,
        "p50_end_to_end_step_seconds": _percentile(e2e_durations, 50),
        "p95_end_to_end_step_seconds": _percentile(e2e_durations, 95),
        "mean_compute_step_seconds": statistics.fmean(compute_durations)
        if compute_durations
        else None,
        "p50_compute_step_seconds": _percentile(compute_durations, 50),
        "p95_compute_step_seconds": _percentile(compute_durations, 95),
        "peak_xpu_allocated_bytes": max(max_allocated) if max_allocated else None,
        "peak_xpu_reserved_bytes": max(max_reserved) if max_reserved else None,
        "minimum_available_system_memory_bytes": min(available) if available else None,
        "maximum_process_rss_bytes": max(process_rss) if process_rss else None,
        "errors": [row["error"] for row in repetitions if row.get("error")],
    }


def estimate_full_workload(
    batch_size: int, train_frames_per_second: float, validation_frames_per_second: float
) -> dict[str, Any]:
    train_frames = 28_800
    validation_frames = 5_600
    epochs = 20
    train_seconds = train_frames / train_frames_per_second
    validation_seconds = validation_frames / validation_frames_per_second
    combined_seconds = train_seconds + validation_seconds
    return {
        "batch_size": batch_size,
        "train_frames_per_epoch": train_frames,
        "validation_frames_per_epoch": validation_frames,
        "train_steps_per_epoch": math.ceil(train_frames / batch_size),
        "validation_steps_per_epoch": math.ceil(validation_frames / batch_size),
        "training_seconds_per_epoch": train_seconds,
        "validation_seconds_per_epoch": validation_seconds,
        "combined_seconds_per_epoch_excluding_checkpoint_overhead": combined_seconds,
        "maximum_20_epoch_seconds_excluding_checkpoint_overhead": combined_seconds * epochs,
        "maximum_training_steps": math.ceil(train_frames / batch_size) * epochs,
        "maximum_validation_steps": math.ceil(validation_frames / batch_size) * epochs,
        "early_stopping_patience": 5,
    }


def recommend_batch(
    batch_results: list[dict[str, Any]], total_system_memory: int
) -> dict[str, Any]:
    minimum_system_headroom = max(int(total_system_memory * 0.20), 4 * GIB)
    safe: list[dict[str, Any]] = []
    for result in batch_results:
        summary = result["training_summary"]
        validation = result["validation_summary"]
        xpu_total = result.get("xpu_total_memory_bytes")
        peak_reserved = summary.get("peak_xpu_reserved_bytes")
        minimum_available = summary.get("minimum_available_system_memory_bytes")
        xpu_headroom_ok = (
            xpu_total is None or peak_reserved is None or peak_reserved <= int(xpu_total * 0.80)
        )
        system_headroom_ok = (
            minimum_available is None or minimum_available >= minimum_system_headroom
        )
        result["headroom"] = {
            "xpu_headroom_ok": xpu_headroom_ok,
            "system_headroom_ok": system_headroom_ok,
            "minimum_required_system_available_bytes": minimum_system_headroom,
        }
        if summary["stable"] and validation["stable"] and xpu_headroom_ok and system_headroom_ok:
            safe.append(result)
    if not safe:
        return {
            "recommended_batch_size": None,
            "reason": "No batch satisfied stability and memory-headroom requirements",
        }

    chosen = max(safe, key=lambda row: int(row["batch_size"]))
    by_batch = {int(row["batch_size"]): row for row in safe}
    if int(chosen["batch_size"]) == 8 and 4 in by_batch:
        batch4 = by_batch[4]["training_summary"]
        batch8 = by_batch[8]["training_summary"]
        fps4 = float(batch4["end_to_end_frames_per_second"])
        fps8 = float(batch8["end_to_end_frames_per_second"])
        improvement = fps8 / fps4 - 1
        reserved4 = batch4.get("peak_xpu_reserved_bytes")
        reserved8 = batch8.get("peak_xpu_reserved_bytes")
        substantial_memory_increase = bool(
            reserved4 and reserved8 and float(reserved8) / float(reserved4) - 1 >= 0.20
        )
        if improvement < 0.10 and substantial_memory_increase:
            chosen = by_batch[4]
            return {
                "recommended_batch_size": 4,
                "reason": (
                    "Batch 8 improved end-to-end throughput by less than 10% while peak "
                    "reserved XPU memory increased by at least 20%"
                ),
                "batch_8_throughput_improvement_fraction": improvement,
            }
    return {
        "recommended_batch_size": int(chosen["batch_size"]),
        "reason": "Largest stable measured batch with required memory headroom",
    }


def _new_trainer(
    model: torch.nn.Module,
    initial_head_state: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
    scratch_checkpoint: Path,
) -> Trainer:
    seed_everything(int(config.get("seed", 42)))
    model.head.load_state_dict(initial_head_state)  # type: ignore[attr-defined]
    train_config = config["train"]
    optimizer = create_optimizer(
        model,
        name=str(train_config.get("optimizer", "adamw")),
        learning_rate=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    return Trainer(
        model,
        optimizer,
        device,
        checkpoint_path=scratch_checkpoint,
        patience=None,
        config=config,
    )


def _run_repetition(
    trainer: Trainer,
    dataset: Dataset[dict[str, object]],
    indices: list[int],
    batch_size: int,
    device: torch.device,
    device_index: int,
    *,
    training: bool,
    repetition: int,
    initial_head_digest: str,
) -> dict[str, Any]:
    loader = _fixed_loader(dataset, indices, batch_size)
    dataset_stats = getattr(dataset, "face_stats", None)
    attempted_before = int(dataset_stats.attempted) if dataset_stats is not None else 0
    failed_before = int(dataset_stats.failed) if dataset_stats is not None else 0
    host_before = _host_memory_snapshot()
    _reset_xpu_peaks(device_index)
    xpu_before = _xpu_memory_snapshot(device_index)
    steps: list[dict[str, Any]] = []
    available_samples = [host_before["available_physical_bytes"]]
    rss_samples = [host_before["process_rss_bytes"]]
    processed_frames = 0
    input_shape = label_shape = None
    error_payload = None
    try:
        iterator = iter(loader)
        for _ in range(math.ceil(len(indices) / batch_size)):
            torch.xpu.synchronize()
            end_to_end_start = time.perf_counter()
            batch = next(iterator)
            torch.xpu.synchronize()
            compute_start = time.perf_counter()
            loss, frames = trainer.run_batch(batch, training=training)
            torch.xpu.synchronize()
            compute_end = time.perf_counter()
            end_to_end_end = compute_end
            processed_frames += frames
            input_shape = list(batch["image"].shape)  # type: ignore[union-attr]
            label_shape = list(batch["label"].shape)  # type: ignore[union-attr]
            steps.append(
                {
                    "frames": frames,
                    "loss": loss,
                    "end_to_end_seconds": end_to_end_end - end_to_end_start,
                    "compute_seconds": compute_end - compute_start,
                }
            )
            host = _host_memory_snapshot()
            available_samples.append(host["available_physical_bytes"])
            rss_samples.append(host["process_rss_bytes"])
    except (FloatingPointError, RuntimeError) as error:
        error_payload = {
            "kind": _failure_kind(error),
            "type": type(error).__name__,
            "message": str(error),
        }
    survived, survival_error = _device_survived(device)
    host_after = _host_memory_snapshot()
    available_samples.append(host_after["available_physical_bytes"])
    rss_samples.append(host_after["process_rss_bytes"])
    attempted_after = int(dataset_stats.attempted) if dataset_stats is not None else 0
    failed_after = int(dataset_stats.failed) if dataset_stats is not None else 0
    return {
        "repetition": repetition,
        "success": error_payload is None and processed_frames == len(indices) and survived,
        "training": training,
        "processed_frames": processed_frames,
        "input_shape": input_shape,
        "label_shape": label_shape,
        "initial_head_digest": initial_head_digest,
        "finite_logits": error_payload is None,
        "finite_loss": error_payload is None,
        "finite_gradients": error_payload is None if training else None,
        "forward_success": error_payload is None,
        "backward_success": error_payload is None if training else None,
        "optimizer_or_scaler_step_success": error_payload is None if training else None,
        "steps": steps,
        "face_detection": {
            "attempted": attempted_after - attempted_before,
            "failed": failed_after - failed_before,
            "failure_rate": (
                (failed_after - failed_before) / (attempted_after - attempted_before)
                if attempted_after > attempted_before
                else 0.0
            ),
        },
        "host_memory_before": host_before,
        "host_memory_after": host_after,
        "available_system_memory_samples_bytes": available_samples,
        "process_rss_samples_bytes": rss_samples,
        "xpu_memory_before": xpu_before,
        "xpu_memory_after": _xpu_memory_snapshot(device_index),
        "device_survived": survived,
        "device_survival_error": survival_error,
        "error": error_payload,
    }


def _warm_up(
    trainer: Trainer,
    dataset: Dataset[dict[str, object]],
    indices: list[int],
    batch_size: int,
    device: torch.device,
    *,
    training: bool,
) -> dict[str, Any]:
    processed = 0
    for batch in _fixed_loader(dataset, indices, batch_size):
        _, frames = trainer.run_batch(batch, training=training)
        processed += frames
    torch.xpu.synchronize()
    survived, error = _device_survived(device)
    return {"frames": processed, "success": processed == WARMUP_FRAMES and survived, "error": error}


def _validate_conditions(config: dict[str, Any]) -> None:
    data = config["data"]
    model = config["model"]
    train = config["train"]
    expected = {
        "frames_per_video": 8,
        "input_resolution": 224,
        "normalization": "clip",
        "frame_sampling_strategy": "uniform",
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"Capacity benchmark requires data.{key}: {value!r}")
    if float(data.get("face_margin", 0.2)) != 0.2:
        raise ValueError("Capacity benchmark requires data.face_margin: 0.2")
    if not bool(model.get("freeze_backbone", True)):
        raise ValueError("Capacity benchmark requires a frozen visual encoder")
    if not bool(train.get("balanced_sampling", False)):
        raise ValueError("Capacity benchmark requires train-only balanced sampling")


def ensure_output_under_runs(output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    runs_root = Path("runs").resolve()
    if not path.is_relative_to(runs_root):
        raise ValueError(f"Capacity report must be written under {runs_root}")
    return path


def run_capacity_benchmark(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    device_request: str = "auto",
) -> dict[str, Any]:
    """Run the bounded batch 2/4/8 XPU benchmark and persist its full evidence."""
    output_path = ensure_output_under_runs(output_path)
    config = load_config(config_path)
    _validate_conditions(config)
    seed = int(config.get("seed", 42))
    device = select_device(device_request)
    if device.type != "xpu":
        raise RuntimeError(f"Capacity benchmark requires Intel XPU, selected: {device}")
    device_index = device.index or 0
    seed_everything(seed)

    reference_train_loader = build_loader(
        manifest_path,
        config["data"],
        split="train",
        batch_size=2,
        balanced_sampling=True,
        seed=seed,
    )
    train_dataset = reference_train_loader.dataset
    balanced_indices = list(reference_train_loader.sampler)
    if len(balanced_indices) < WARMUP_FRAMES + MEASURED_FRAMES:
        raise ValueError("Benchmark train subset must provide at least 40 frame samples")
    balanced_indices = balanced_indices[: WARMUP_FRAMES + MEASURED_FRAMES]

    reference_validation_loader = build_loader(
        manifest_path,
        config["data"],
        split="validation",
        batch_size=2,
        balanced_sampling=False,
        seed=seed,
    )
    validation_dataset = reference_validation_loader.dataset
    validation_indices = list(range(len(validation_dataset)))
    if len(validation_indices) < WARMUP_FRAMES + MEASURED_FRAMES:
        raise ValueError("Benchmark validation subset must provide at least 40 frame samples")
    validation_indices = validation_indices[: WARMUP_FRAMES + MEASURED_FRAMES]

    model = build_model(config["model"]).to(device)
    trainable_encoder_parameters = sum(
        parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad
    )
    if trainable_encoder_parameters:
        raise RuntimeError("Capacity benchmark detected a trainable visual encoder")
    initial_head_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.head.state_dict().items()
    }
    initial_head_digest = _state_digest(initial_head_state)
    properties = torch.xpu.get_device_properties(device_index)
    host_at_start = _host_memory_snapshot()
    scratch_checkpoint = output_path.parent / "benchmark-not-saved.pt"
    batch_results: list[dict[str, Any]] = []

    for batch_size in BATCH_SIZES:
        warmup_trainer = _new_trainer(model, initial_head_state, config, device, scratch_checkpoint)
        warmup = _warm_up(
            warmup_trainer,
            train_dataset,
            balanced_indices[:WARMUP_FRAMES],
            batch_size,
            device,
            training=True,
        )
        training_repetitions: list[dict[str, Any]] = []
        validation_repetitions: list[dict[str, Any]] = []
        if warmup["success"]:
            validation_warmup_trainer = _new_trainer(
                model, initial_head_state, config, device, scratch_checkpoint
            )
            validation_warmup = _warm_up(
                validation_warmup_trainer,
                validation_dataset,
                validation_indices[:WARMUP_FRAMES],
                batch_size,
                device,
                training=False,
            )
            for repetition in range(1, REPETITIONS + 1):
                trainer = _new_trainer(
                    model, initial_head_state, config, device, scratch_checkpoint
                )
                digest = _state_digest(model.head.state_dict())
                training_repetitions.append(
                    _run_repetition(
                        trainer,
                        train_dataset,
                        balanced_indices[WARMUP_FRAMES:],
                        batch_size,
                        device,
                        device_index,
                        training=True,
                        repetition=repetition,
                        initial_head_digest=digest,
                    )
                )
                validation_trainer = _new_trainer(
                    model, initial_head_state, config, device, scratch_checkpoint
                )
                validation_digest = _state_digest(model.head.state_dict())
                validation_repetitions.append(
                    _run_repetition(
                        validation_trainer,
                        validation_dataset,
                        validation_indices[WARMUP_FRAMES:],
                        batch_size,
                        device,
                        device_index,
                        training=False,
                        repetition=repetition,
                        initial_head_digest=validation_digest,
                    )
                )
        else:
            validation_warmup = {"frames": 0, "success": False, "error": "training warm-up failed"}

        training_summary = summarize_repetitions(training_repetitions)
        validation_summary = summarize_repetitions(validation_repetitions)
        validation_summary["validation_frames_per_second"] = validation_summary.pop(
            "end_to_end_frames_per_second"
        )
        batch_results.append(
            {
                "batch_size": batch_size,
                "expected_input_shape": [batch_size, 3, 224, 224],
                "expected_label_shape": [batch_size],
                "sample_sequence_digest": _indices_digest(balanced_indices),
                "initial_head_digest": initial_head_digest,
                "xpu_total_memory_bytes": int(properties.total_memory),
                "training_warmup": warmup,
                "validation_warmup": validation_warmup,
                "training_repetitions": training_repetitions,
                "validation_repetitions": validation_repetitions,
                "training_summary": training_summary,
                "validation_summary": validation_summary,
            }
        )
        if not warmup["success"] or any(
            row.get("error", {}).get("kind")
            in {"xpu_oom", "allocation_failure", "device_or_driver_error"}
            for row in training_repetitions
            if row.get("error")
        ):
            break

    total_system_memory = int(host_at_start.get("total_physical_bytes") or 0)
    recommendation = recommend_batch(batch_results, total_system_memory)
    recommended_batch = recommendation.get("recommended_batch_size")
    workload = None
    if recommended_batch is not None:
        result = next(row for row in batch_results if row["batch_size"] == recommended_batch)
        train_fps = result["training_summary"].get("end_to_end_frames_per_second")
        validation_fps = result["validation_summary"].get("validation_frames_per_second")
        if train_fps and validation_fps:
            workload = estimate_full_workload(recommended_batch, train_fps, validation_fps)

    report = {
        "schema_version": 1,
        "artifact_type": "deeptector_m1_xpu_capacity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "xpu_available": torch.xpu.is_available(),
            "device": str(device),
            "device_name": torch.xpu.get_device_name(device_index),
            "device_properties": str(properties),
            "xpu_total_memory_bytes": int(properties.total_memory),
            "host_memory_at_start": host_at_start,
            "amp_enabled": True,
            "amp_dtype": "float16",
            "amp_initial_scale": float(config["train"].get("amp_initial_scale", 65536.0)),
        },
        "conditions": {
            "config": str(config_path),
            "manifest": str(manifest_path),
            "seed": seed,
            "batch_sizes": list(BATCH_SIZES),
            "warmup_frames": WARMUP_FRAMES,
            "measured_frames_per_repetition": MEASURED_FRAMES,
            "repetitions": REPETITIONS,
            "frames_per_video": int(config["data"]["frames_per_video"]),
            "input_resolution": int(config["data"]["input_resolution"]),
            "face_margin": float(config["data"]["face_margin"]),
            "normalization": str(config["data"]["normalization"]),
            "balanced_sampling": True,
            "trainable_encoder_parameters": trainable_encoder_parameters,
            "batch_semantics": "independent flattened frame samples entering CLIP",
            "compute_timing_boundary": (
                "preprocessed host batch ready through H2D, forward, loss, backward, "
                "and optimizer/scaler step"
            ),
            "end_to_end_timing_boundary": (
                "video decode, face crop, preprocessing, collate, H2D, model, loss, "
                "backward, and optimizer/scaler step"
            ),
            "initial_state_equivalence": (
                "One deterministic head state was cloned before measurement and reloaded before "
                "every repetition; the frozen CLIP instance was unchanged."
            ),
        },
        "batch_results": batch_results,
        "recommendation": recommendation,
        "full_workload_estimate": workload,
        "estimate_uncertainty": (
            "Bounded 32-frame repetitions include the real synchronous preprocessing path, but "
            "long-run thermal throttling, OS contention, checkpoint I/O, and driver behavior may "
            "increase full-experiment wall time."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
