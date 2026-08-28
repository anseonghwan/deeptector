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
