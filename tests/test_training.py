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
