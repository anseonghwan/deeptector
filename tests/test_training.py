from types import SimpleNamespace

import pytest
import torch
from helpers import TinyEncoder
from torch.utils.data import DataLoader, Dataset

from deeptector.models.classifier import DeepfakeClassifier
from deeptector.training.checkpoint import load_checkpoint, save_checkpoint
from deeptector.training.trainer import Trainer, create_optimizer


class SyntheticDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        label = float(index % 2)
        return {"image": torch.full((3, 8, 8), label), "label": torch.tensor(label)}


class DirectLogitModel(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.mode = mode

    def forward(self, images):
        logits = self.weight.expand(images.shape[0])
        if self.mode == "logits":
            logits = logits * torch.tensor(float("inf"), device=logits.device)
        elif self.mode == "gradient":
            logits = NonFiniteGradient.apply(logits)
        return SimpleNamespace(logits=logits)


class NonFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        return values.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class NonFiniteLabelDataset(SyntheticDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        row["label"] = torch.tensor(float("nan"))
        return row


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def test_checkpoint_round_trip(tmp_path):
    model = DeepfakeClassifier(TinyEncoder())
    optimizer = create_optimizer(model)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=3, best_metric=0.7)
    metadata = load_checkpoint(path, model, optimizer)
    assert metadata["epoch"] == 3
    assert metadata["best_metric"] == 0.7


def test_synthetic_training_smoke(tmp_path):
    torch.manual_seed(4)
    model = DeepfakeClassifier(TinyEncoder())
    optimizer = create_optimizer(model, learning_rate=0.01)
    loader = DataLoader(SyntheticDataset(), batch_size=4)
    trainer = Trainer(
        model, optimizer, torch.device("cpu"), checkpoint_path=tmp_path / "best.pt", patience=2
    )
    result = trainer.fit(loader, loader, epochs=2)
    assert result.epochs_completed == 2
    assert (tmp_path / "best.pt").exists()


def _trainer_for_failure(model, loader, tmp_path):
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    checkpoint = tmp_path / "failed.pt"
    trainer = Trainer(model, optimizer, torch.device("cpu"), checkpoint_path=checkpoint)
    return trainer, optimizer, checkpoint, loader


def test_training_rejects_non_finite_logits_before_step(tmp_path):
    loader = DataLoader(SyntheticDataset(), batch_size=2)
    trainer, optimizer, checkpoint, loader = _trainer_for_failure(
        DirectLogitModel("logits"), loader, tmp_path
    )

    with pytest.raises(FloatingPointError, match="Non-finite logits detected during training"):
        trainer.fit(loader, loader, epochs=1)

    assert optimizer.step_calls == 0
    assert not checkpoint.exists()
    assert not (tmp_path / "last_checkpoint.pt").exists()


def test_training_rejects_non_finite_loss_before_step(tmp_path):
    loader = DataLoader(NonFiniteLabelDataset(), batch_size=2)
    trainer, optimizer, checkpoint, loader = _trainer_for_failure(
        DirectLogitModel("loss"), loader, tmp_path
    )

    with pytest.raises(FloatingPointError, match="Non-finite training loss detected"):
        trainer.fit(loader, loader, epochs=1)

    assert optimizer.step_calls == 0
    assert not checkpoint.exists()
    assert not (tmp_path / "last_checkpoint.pt").exists()


def test_training_rejects_non_finite_gradient_before_step(tmp_path):
    loader = DataLoader(SyntheticDataset(), batch_size=2)
    trainer, optimizer, checkpoint, loader = _trainer_for_failure(
        DirectLogitModel("gradient"), loader, tmp_path
    )

    with pytest.raises(FloatingPointError, match="Non-finite gradient detected in parameter"):
        trainer.fit(loader, loader, epochs=1)

    assert optimizer.step_calls == 0
    assert not checkpoint.exists()
    assert not (tmp_path / "last_checkpoint.pt").exists()


@torch.no_grad()
def _xpu_outputs_are_finite(model, loader):
    batch = next(iter(loader))
    output = model(batch["image"].to("xpu"))
    return bool(torch.isfinite(output.logits).all())


@pytest.mark.skipif(
    not (getattr(torch, "xpu", None) and torch.xpu.is_available()),
    reason="Intel XPU is unavailable",
)
def test_xpu_training_smoke(tmp_path):
    torch.manual_seed(4)
    torch.xpu.manual_seed_all(4)
    device = torch.device("xpu")
    model = DeepfakeClassifier(TinyEncoder())
    optimizer = create_optimizer(model, learning_rate=0.01)
    loader = DataLoader(SyntheticDataset(), batch_size=2)
    trainer = Trainer(
        model,
        optimizer,
        device,
        checkpoint_path=tmp_path / "xpu-best.pt",
        patience=1,
    )

    result = trainer.fit(loader, loader, epochs=1)

    assert trainer.amp_enabled
    assert next(model.parameters()).device.type == "xpu"
    assert result.epochs_completed == 1
    assert (tmp_path / "xpu-best.pt").exists()
    assert _xpu_outputs_are_finite(model, loader)
