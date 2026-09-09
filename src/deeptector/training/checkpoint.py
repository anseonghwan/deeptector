"""Portable checkpoint persistence."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CHECKPOINT_SCHEMA_VERSION = 2


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    best_metric: float,
    config: dict[str, Any] | None = None,
) -> None:
    """Atomically save model and training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": "best",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config or {},
        },
        temporary,
    )
    temporary.replace(path)


def capture_rng_state() -> dict[str, Any]:
    """Capture process RNG state required for deterministic epoch-boundary resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "cuda": None,
        "xpu": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        state["xpu"] = xpu.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu", "cuda", "xpu"}
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError(f"Resume checkpoint RNG state is missing: {', '.join(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Resume checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if state["xpu"] is not None:
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise RuntimeError("Resume checkpoint contains XPU RNG state but XPU is unavailable")
        xpu.set_rng_state_all([value.cpu() for value in state["xpu"]])


def save_resume_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    completed_epoch: int,
    best_validation_loss: float,
    best_epoch: int,
    stale_epochs: int,
    config: dict[str, Any],
    sampler_generator_state: torch.Tensor | None,
    device_type: str,
) -> None:
    """Atomically save the latest successful epoch boundary for training resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": "last",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "training_state": {
                "completed_epoch": completed_epoch,
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "device_type": device_type,
            },
            "rng_state": capture_rng_state(),
            "sampler_generator_state": sampler_generator_state,
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def load_resume_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    expected_config: dict[str, Any],
    expected_device_type: str,
) -> dict[str, Any]:
    """Load and validate a full epoch-boundary training checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Resume checkpoint schema is incompatible with this training version")
    if payload.get("checkpoint_kind") != "last":
        raise ValueError("Training resume requires a last_checkpoint.pt resume checkpoint")
    required = {
        "model",
        "optimizer",
        "scaler",
        "training_state",
        "rng_state",
        "sampler_generator_state",
        "config",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Resume checkpoint is missing required state: {', '.join(missing)}")
    if payload["config"] != expected_config:
        raise ValueError("Resume checkpoint config does not match the requested experiment config")
    training_state = payload["training_state"]
    required_training = {
        "completed_epoch",
        "best_validation_loss",
        "best_epoch",
        "stale_epochs",
        "device_type",
    }
    missing_training = sorted(required_training - training_state.keys())
    if missing_training:
        raise ValueError(
            "Resume checkpoint training state is missing: " + ", ".join(missing_training)
        )
    if training_state["device_type"] != expected_device_type:
        raise ValueError(
            "Resume checkpoint device type does not match the requested training device: "
            f"{training_state['device_type']} != {expected_device_type}"
        )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng_state"])
    return {
        "training_state": training_state,
        "sampler_generator_state": payload["sampler_generator_state"],
    }


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
    expected_kind: str | None = None,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate, load state, and return checkpoint metadata."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if expected_kind is not None:
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Checkpoint schema is incompatible with this training version")
        if payload.get("checkpoint_kind") != expected_kind:
            raise ValueError(f"Evaluation requires a {expected_kind} checkpoint")
    if expected_config is not None and payload.get("config") != expected_config:
        raise ValueError("Checkpoint config does not match the requested experiment config")
    if "model" not in payload:
        raise ValueError("Checkpoint is missing model state")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return {key: value for key, value in payload.items() if key not in {"model", "optimizer"}}
