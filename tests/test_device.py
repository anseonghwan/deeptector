import pytest
import torch

from deeptector.utils.device import select_device


def test_auto_prefers_cuda_over_xpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)

    assert select_device("auto") == torch.device("cuda")


def test_auto_uses_xpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)

    assert select_device("auto") == torch.device("xpu")


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)

    assert select_device("auto") == torch.device("cpu")


def test_explicit_xpu_is_rejected_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="XPU was requested but is unavailable"):
        select_device("xpu")


def test_explicit_xpu_preserves_device_index(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)

    assert select_device("xpu:0") == torch.device("xpu:0")
