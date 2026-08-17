"""Minimal deterministic PyTorch trainer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .losses import binary_classification_loss

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Training outcome and best validation state."""

    best_epoch: int
    best_validation_loss: float
    epochs_completed: int


class Trainer:
    """Single-device trainer with AMP, early stopping, and best checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        checkpoint_path: str | Path,
        patience: int | None = 5,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.patience = patience
        self.config = config or {}
        self.loss_fn = binary_classification_loss()
        self.amp_enabled = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def fit(
        self,
        train_loader: DataLoader[dict[str, object]],
        validation_loader: DataLoader[dict[str, object]],
        epochs: int,
    ) -> TrainResult:
        """Train and retain the checkpoint with minimum validation loss."""
        if epochs < 1:
            raise ValueError("epochs must be positive")
        best_loss, best_epoch, stale = float("inf"), -1, 0
        completed = 0
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True)
            validation_loss = self._run_epoch(validation_loader, training=False)
            completed = epoch
            LOGGER.info(
                "epoch=%d train_loss=%.6f validation_loss=%.6f", epoch, train_loss, validation_loss
            )
            if validation_loss < best_loss:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                save_checkpoint(
                    self.checkpoint_path,
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    best_metric=-validation_loss,
                    config=self.config,
                )
            else:
                stale += 1
                if self.patience is not None and stale >= self.patience:
                    LOGGER.info("early_stopping epoch=%d patience=%d", epoch, self.patience)
                    break
        self._log_face_stats("train", train_loader)
        self._log_face_stats("validation", validation_loader)
        return TrainResult(best_epoch, best_loss, completed)

    @staticmethod
    def _log_face_stats(name: str, loader: DataLoader[dict[str, object]]) -> None:
        stats = getattr(loader.dataset, "face_stats", None)
        if stats is not None:
            LOGGER.info(
                "%s_face_detection attempted=%d failed=%d failure_rate=%.6f",
                name,
                stats.attempted,
                stats.failed,
                stats.failure_rate,
            )

    def _run_epoch(self, loader: DataLoader[dict[str, object]], *, training: bool) -> float:
        self.model.train(training)
        total_loss, samples = 0.0, 0
        context = torch.enable_grad if training else torch.no_grad
        with context():
            for batch in loader:
                images = batch["image"].to(self.device)  # type: ignore[union-attr]
                labels = batch["label"].to(self.device)  # type: ignore[union-attr]
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    output = self.model(images)
                    loss = self.loss_fn(output.logits, labels)
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                batch_size = int(labels.shape[0])
                total_loss += float(loss.detach()) * batch_size
                samples += batch_size
        if samples == 0:
            raise ValueError("DataLoader yielded no samples")
        return total_loss / samples


def create_optimizer(
    model: nn.Module,
    *,
    name: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build a configured optimizer over trainable parameters only."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    if name.lower() == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name.lower() == "sgd":
        return torch.optim.SGD(
            parameters, lr=learning_rate, weight_decay=weight_decay, momentum=0.9
        )
    raise ValueError(f"Unsupported optimizer: {name}")
