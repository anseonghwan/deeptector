import copy
import random

import numpy as np
import pytest
import torch
from helpers import TinyEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from deeptector.models.classifier import DeepfakeClassifier
from deeptector.training.checkpoint import (
    load_resume_checkpoint,
    save_resume_checkpoint,
)
from deeptector.training.trainer import Trainer, create_optimizer
from deeptector.utils.seed import seed_everything


class RecordingDataset(Dataset):
    def __init__(self):
        self.indices = []

    def __len__(self):
        return 8

    def __getitem__(self, index):
        self.indices.append(index)
        label = float(index % 2)
        return {
            "image": torch.full((3, 8, 8), label),
            "label": torch.tensor(label),
        }


class ScriptedTrainer(Trainer):
    def __init__(self, *args, validation_losses, fail_training_epoch=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_losses = iter(validation_losses)
        self.training_epoch = 0
        self.fail_training_epoch = fail_training_epoch

    def _run_epoch(self, loader, *, training):
        if training:
            self.training_epoch += 1
            if self.training_epoch == self.fail_training_epoch:
                raise FloatingPointError("scripted numerical failure")
            return 0.25
        return next(self.validation_losses)


def _balanced_loader(dataset, *, seed=42):
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        [1.0] * len(dataset),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=2, sampler=sampler)


def _validation_loader():
    return DataLoader(RecordingDataset(), batch_size=2)


def _config(*, amp=False):
    return {
        "seed": 23,
        "model": {"name": "tiny", "freeze_backbone": False},
        "train": {
            "epochs": 3,
            "batch_size": 2,
            "balanced_sampling": True,
            "amp": amp,
            "amp_initial_scale": 128,
            "optimizer": "adamw",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "early_stopping_patience": 5,
        },
    }


def _new_training(directory, *, device=None, config=None):
    device = device or torch.device("cpu")
    model = DeepfakeClassifier(TinyEncoder())
    optimizer = create_optimizer(model, learning_rate=0.01, weight_decay=0.0)
    trainer = Trainer(
        model,
        optimizer,
        device,
        checkpoint_path=directory / "checkpoint.pt",
        patience=5,
        config=config or _config(),
    )
    return model, optimizer, trainer


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path):
    config = _config()

    seed_everything(23)
    continuous_dataset = RecordingDataset()
    continuous_loader = _balanced_loader(continuous_dataset)
    model, optimizer, trainer = _new_training(tmp_path / "continuous", config=config)
    continuous_result = trainer.fit(
        continuous_loader,
        _validation_loader(),
        epochs=3,
    )
    continuous_model = copy.deepcopy(model.state_dict())
    continuous_optimizer = copy.deepcopy(optimizer.state_dict())
    continuous_scaler = copy.deepcopy(trainer.scaler.state_dict())
    continuous_rng_draws = (random.random(), np.random.random(), torch.rand(3))
    continuous_last = torch.load(
        trainer.last_checkpoint_path, map_location="cpu", weights_only=False
    )

    seed_everything(23)
    first_dataset = RecordingDataset()
    first_loader = _balanced_loader(first_dataset)
    _, _, first_trainer = _new_training(tmp_path / "resumed", config=config)
    first_result = first_trainer.fit(first_loader, _validation_loader(), epochs=2)
    assert first_result.epochs_completed == 2

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed_dataset = RecordingDataset()
    resumed_loader = _balanced_loader(resumed_dataset)
    resumed_model, resumed_optimizer, resumed_trainer = _new_training(
        tmp_path / "resumed", config=config
    )
    resume_state = resumed_trainer.load_resume_state(
        first_trainer.last_checkpoint_path,
        resumed_loader,
    )
    assert resume_state.completed_epoch == 2
    resumed_result = resumed_trainer.fit(
        resumed_loader,
        _validation_loader(),
        epochs=3,
        resume_state=resume_state,
    )
    resumed_rng_draws = (random.random(), np.random.random(), torch.rand(3))
    resumed_last = torch.load(
        resumed_trainer.last_checkpoint_path, map_location="cpu", weights_only=False
    )

    assert continuous_result.epochs_completed == resumed_result.epochs_completed == 3
    assert continuous_dataset.indices == first_dataset.indices + resumed_dataset.indices
    _assert_nested_equal(continuous_model, resumed_model.state_dict())
    _assert_nested_equal(continuous_optimizer, resumed_optimizer.state_dict())
    _assert_nested_equal(continuous_scaler, resumed_trainer.scaler.state_dict())
    _assert_nested_equal(continuous_last["training_state"], resumed_last["training_state"])
    _assert_nested_equal(
        continuous_last["sampler_generator_state"],
        resumed_last["sampler_generator_state"],
    )
    assert continuous_rng_draws[0] == resumed_rng_draws[0]
    assert continuous_rng_draws[1] == resumed_rng_draws[1]
    assert torch.equal(continuous_rng_draws[2], resumed_rng_draws[2])


def test_non_improving_epoch_updates_last_but_not_best(tmp_path):
    model = DeepfakeClassifier(TinyEncoder())
    optimizer = create_optimizer(model)
    trainer = ScriptedTrainer(
        model,
        optimizer,
        torch.device("cpu"),
        checkpoint_path=tmp_path / "checkpoint.pt",
        patience=5,
        config={"train": {"balanced_sampling": False}},
        validation_losses=[0.4, 0.6],
    )
    loader = DataLoader(RecordingDataset(), batch_size=2)

    result = trainer.fit(loader, loader, epochs=2)
    best = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=False)
    last = torch.load(trainer.last_checkpoint_path, map_location="cpu", weights_only=False)

    assert result.best_epoch == 1
    assert best["checkpoint_kind"] == "best"
    assert best["epoch"] == 1
    assert last["checkpoint_kind"] == "last"
    assert last["training_state"] == {
        "completed_epoch": 2,
        "best_validation_loss": 0.4,
        "best_epoch": 1,
        "stale_epochs": 1,
        "device_type": "cpu",
    }


def test_improving_epoch_updates_best_checkpoint(tmp_path):
    model = DeepfakeClassifier(TinyEncoder())
    optimizer = create_optimizer(model)
    trainer = ScriptedTrainer(
        model,
        optimizer,
        torch.device("cpu"),
        checkpoint_path=tmp_path / "checkpoint.pt",
        config={"train": {"balanced_sampling": False}},
        validation_losses=[0.5, 0.3],
    )
    loader = DataLoader(RecordingDataset(), batch_size=2)

    trainer.fit(loader, loader, epochs=2)
    best = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=False)
    last = torch.load(trainer.last_checkpoint_path, map_location="cpu", weights_only=False)

    assert best["epoch"] == 2
    assert best["best_metric"] == -0.3
    assert last["training_state"]["best_epoch"] == 2
    assert last["training_state"]["stale_epochs"] == 0


def test_failed_epoch_does_not_replace_last_successful_boundary(tmp_path):
    model = DeepfakeClassifier(TinyEncoder())
    optimizer = create_optimizer(model)
    trainer = ScriptedTrainer(
        model,
        optimizer,
        torch.device("cpu"),
        checkpoint_path=tmp_path / "checkpoint.pt",
        config={"train": {"balanced_sampling": False}},
        validation_losses=[0.5],
        fail_training_epoch=2,
    )
    loader = DataLoader(RecordingDataset(), batch_size=2)

    with pytest.raises(FloatingPointError, match="scripted numerical failure"):
        trainer.fit(loader, loader, epochs=2)

    last = torch.load(trainer.last_checkpoint_path, map_location="cpu", weights_only=False)
    assert last["training_state"]["completed_epoch"] == 1


def test_resume_preserves_early_stopping_progress(tmp_path):
    config = {
        "train": {
            "epochs": 3,
            "balanced_sampling": False,
            "early_stopping_patience": 2,
        }
    }
    loader = DataLoader(RecordingDataset(), batch_size=2)
    first_model = DeepfakeClassifier(TinyEncoder())
    first_optimizer = create_optimizer(first_model)
    first_trainer = ScriptedTrainer(
        first_model,
        first_optimizer,
        torch.device("cpu"),
        checkpoint_path=tmp_path / "checkpoint.pt",
        patience=2,
        config=config,
        validation_losses=[0.4, 0.5],
    )
    first_trainer.fit(loader, loader, epochs=2)

    resumed_model = DeepfakeClassifier(TinyEncoder())
    resumed_optimizer = create_optimizer(resumed_model)
    resumed_trainer = ScriptedTrainer(
        resumed_model,
        resumed_optimizer,
        torch.device("cpu"),
        checkpoint_path=tmp_path / "checkpoint.pt",
        patience=2,
        config=config,
        validation_losses=[0.6],
    )
    state = resumed_trainer.load_resume_state(first_trainer.last_checkpoint_path, loader)
    assert state.stale_epochs == 1

    result = resumed_trainer.fit(loader, loader, epochs=3, resume_state=state)
    last = torch.load(resumed_trainer.last_checkpoint_path, map_location="cpu", weights_only=False)

    assert result.epochs_completed == 3
    assert result.best_epoch == 1
    assert last["training_state"]["stale_epochs"] == 2


def test_resume_rejects_incompatible_config_before_loading(tmp_path):
    config = _config()
    seed_everything(23)
    dataset = RecordingDataset()
    loader = _balanced_loader(dataset)
    _, _, trainer = _new_training(tmp_path, config=config)
    trainer.fit(loader, _validation_loader(), epochs=1)

    incompatible = copy.deepcopy(config)
    incompatible["train"]["batch_size"] = 8
    resumed_dataset = RecordingDataset()
    resumed_loader = _balanced_loader(resumed_dataset)
    _, _, resumed_trainer = _new_training(tmp_path, config=incompatible)

    with pytest.raises(ValueError, match="config does not match"):
        resumed_trainer.load_resume_state(trainer.last_checkpoint_path, resumed_loader)


def test_scaler_state_round_trip_in_resume_checkpoint(tmp_path):
    seed_everything(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=128)
    inputs = torch.ones(2, 2)
    loss = model(inputs).sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    expected_scaler = copy.deepcopy(scaler.state_dict())
    config = {"train": {"epochs": 2}}
    sampler_state = torch.Generator().manual_seed(4).get_state()
    path = tmp_path / "last_checkpoint.pt"
    save_resume_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        completed_epoch=1,
        best_validation_loss=0.5,
        best_epoch=1,
        stale_epochs=0,
        config=config,
        sampler_generator_state=sampler_state,
        device_type="cpu",
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1)
    metadata = load_resume_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        restored_scaler,
        expected_config=config,
        expected_device_type="cpu",
    )

    _assert_nested_equal(model.state_dict(), restored_model.state_dict())
    _assert_nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    assert restored_scaler.state_dict() == expected_scaler
    assert metadata["training_state"]["completed_epoch"] == 1
    assert torch.equal(metadata["sampler_generator_state"], sampler_state)


@pytest.mark.skipif(
    not (getattr(torch, "xpu", None) and torch.xpu.is_available()),
    reason="Intel XPU is unavailable",
)
def test_xpu_epoch_boundary_resume_smoke(tmp_path):
    config = _config(amp=True)
    config["train"]["epochs"] = 2
    device = torch.device("xpu")
    seed_everything(23)
    first_dataset = RecordingDataset()
    first_loader = _balanced_loader(first_dataset)
    _, _, first_trainer = _new_training(tmp_path, device=device, config=config)
    first_trainer.fit(first_loader, _validation_loader(), epochs=1)
    first_payload = torch.load(
        first_trainer.last_checkpoint_path, map_location="cpu", weights_only=False
    )

    resumed_dataset = RecordingDataset()
    resumed_loader = _balanced_loader(resumed_dataset)
    resumed_model, _, resumed_trainer = _new_training(tmp_path, device=device, config=config)
    state = resumed_trainer.load_resume_state(first_trainer.last_checkpoint_path, resumed_loader)

    assert state.completed_epoch == 1
    assert resumed_trainer.scaler.state_dict() == first_payload["scaler"]
    result = resumed_trainer.fit(
        resumed_loader,
        _validation_loader(),
        epochs=2,
        resume_state=state,
    )
    output = resumed_model(torch.ones(2, 3, 8, 8, device=device))
    torch.xpu.synchronize()
    final_payload = torch.load(
        resumed_trainer.last_checkpoint_path, map_location="cpu", weights_only=False
    )

    assert result.epochs_completed == 2
    assert final_payload["training_state"]["completed_epoch"] == 2
    assert torch.isfinite(output.logits).all()
