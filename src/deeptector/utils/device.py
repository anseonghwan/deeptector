"""Compute-device selection."""

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda and reject unavailable CUDA explicitly."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device
