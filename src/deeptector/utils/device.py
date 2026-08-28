"""Compute-device selection."""

import torch


def is_xpu_available() -> bool:
    """Return whether this PyTorch build can use an Intel XPU device."""
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and bool(xpu.is_available())


def select_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda/xpu and reject unavailable accelerators."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_xpu_available():
            return torch.device("xpu")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "xpu" and not is_xpu_available():
        raise RuntimeError("XPU was requested but is unavailable")
    return device
