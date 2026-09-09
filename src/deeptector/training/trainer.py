"""Minimal deterministic PyTorch trainer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import load_resume_checkpoint, save_checkpoint, save_resume_checkpoint
from .losses import binary_classification_loss

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Training outcome and best validation state."""

    best_epoch: int
    best_validation_loss: float
    epochs_completed: int


@dataclass(frozen=True)
class ResumeState:
    """State restored from the latest successfully completed epoch."""

    completed_epoch: int
    best_validation_loss: float
    best_epoch: int
    stale_epochs: int


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
        self.last_checkpoint_path = self.checkpoint_path.with_name("last_checkpoint.pt")
        self.patience = patience
        self.config = config or {}
        self.loss_fn = binary_classification_loss()
        amp_requested = bool(self.config.get("train", {}).get("amp", True))
        amp_initial_scale = float(self.config.get("train", {}).get("amp_initial_scale", 65536.0))
        self.amp_enabled = amp_requested and device.type in {"cuda", "xpu"}
        self.scaler = torch.amp.GradScaler(
            device.type,
            enabled=self.amp_enabled,
            init_scale=amp_initial_scale,
        )
        self.optimizer_steps = 0
        self.amp_overflow_skips = 0

    def fit(
        self,
        train_loader: DataLoader[dict[str, object]],
        validation_loader: DataLoader[dict[str, object]],
        epochs: int,
        *,
        resume_state: ResumeState | None = None,
    ) -> TrainResult:
        """Train and retain the checkpoint with minimum validation loss."""
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if resume_state is None:
            best_loss, best_epoch, stale, completed = float("inf"), -1, 0, 0
        else:
            best_loss = resume_state.best_validation_loss
            best_epoch = resume_state.best_epoch
            stale = resume_state.stale_epochs
            completed = resume_state.completed_epoch
        if completed > epochs:
            raise ValueError(
                f"Resume checkpoint completed epoch {completed}, beyond configured maximum {epochs}"
            )
        if self.patience is not None and stale >= self.patience:
            LOGGER.info(
                "resume_checkpoint_already_early_stopped epoch=%d stale=%d patience=%d",
                completed,
                stale,
                self.patience,
            )
            return TrainResult(best_epoch, best_loss, completed)
        for epoch in range(completed + 1, epochs + 1):
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
            save_resume_checkpoint(
                self.last_checkpoint_path,
                self.model,
                self.optimizer,
                self.scaler,
                completed_epoch=completed,
                best_validation_loss=best_loss,
                best_epoch=best_epoch,
                stale_epochs=stale,
                config=self.config,
                sampler_generator_state=self._sampler_generator_state(train_loader),
                device_type=self.device.type,
            )
            if self.patience is not None and stale >= self.patience:
                LOGGER.info("early_stopping epoch=%d patience=%d", epoch, self.patience)
                break
        self._log_face_stats("train", train_loader)
        self._log_face_stats("validation", validation_loader)
        return TrainResult(best_epoch, best_loss, completed)

    def load_resume_state(
        self,
        path: str | Path,
        train_loader: DataLoader[dict[str, object]],
    ) -> ResumeState:
        """Restore a successful epoch boundary and its deterministic sampler position."""
        metadata = load_resume_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scaler,
            expected_config=self.config,
            expected_device_type=self.device.type,
        )
        training_state = metadata["training_state"]
        state = ResumeState(
            completed_epoch=int(training_state["completed_epoch"]),
            best_validation_loss=float(training_state["best_validation_loss"]),
            best_epoch=int(training_state["best_epoch"]),
            stale_epochs=int(training_state["stale_epochs"]),
        )
        if state.completed_epoch < 1:
            raise ValueError("Resume checkpoint completed_epoch must be positive")
        if not 1 <= state.best_epoch <= state.completed_epoch:
            raise ValueError("Resume checkpoint best_epoch is outside the completed epoch range")
        if state.stale_epochs < 0:
            raise ValueError("Resume checkpoint stale_epochs must not be negative")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Best evaluation checkpoint is missing beside resume state: {self.checkpoint_path}"
            )
        self._restore_sampler_generator_state(train_loader, metadata["sampler_generator_state"])
        return state

    def _sampler_generator_state(
        self, loader: DataLoader[dict[str, object]]
    ) -> torch.Tensor | None:
        generator = getattr(loader.sampler, "generator", None)
        if generator is None:
            if bool(self.config.get("train", {}).get("balanced_sampling", False)):
                raise RuntimeError("Balanced training sampler has no restorable generator")
            return None
        return generator.get_state()

    def _restore_sampler_generator_state(
        self,
        loader: DataLoader[dict[str, object]],
        state: torch.Tensor | None,
    ) -> None:
        generator = getattr(loader.sampler, "generator", None)
        if state is None:
            if bool(self.config.get("train", {}).get("balanced_sampling", False)):
                raise ValueError("Resume checkpoint has no balanced-sampler generator state")
            return
        if generator is None:
            raise ValueError("Current training loader has no generator for saved sampler state")
        generator.set_state(state.cpu())

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
        for batch in loader:
            loss, batch_size = self.run_batch(batch, training=training)
            total_loss += loss * batch_size
            samples += batch_size
        if samples == 0:
            raise ValueError("DataLoader yielded no samples")
        return total_loss / samples

    def run_batch(self, batch: dict[str, object], *, training: bool) -> tuple[float, int]:
        """Run one collated batch through the same guarded train/validation path."""
        self.model.train(training)
        images = batch["image"].to(self.device)  # type: ignore[union-attr]
        labels = batch["label"].to(self.device)  # type: ignore[union-attr]
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad if training else torch.no_grad
        with context():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                output = self.model(images)
                phase = "training" if training else "validation"
                if not torch.isfinite(output.logits).all():
                    raise FloatingPointError(f"Non-finite logits detected during {phase}")
                loss = self.loss_fn(output.logits, labels)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(f"Non-finite {phase} loss detected")
            if training:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                non_finite_parameter = self._first_non_finite_gradient()
                if non_finite_parameter is not None and not self.amp_enabled:
                    raise FloatingPointError(
                        f"Non-finite gradient detected in parameter: {non_finite_parameter}"
                    )
                scale_before = float(self.scaler.get_scale())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if non_finite_parameter is not None:
                    self.amp_overflow_skips += 1
                    scale_after = float(self.scaler.get_scale())
                    if scale_after >= scale_before:
                        raise FloatingPointError(
                            "AMP did not reduce the loss scale after a non-finite gradient in "
                            f"parameter: {non_finite_parameter}"
                        )
                    LOGGER.warning(
                        "amp_gradient_overflow parameter=%s optimizer_step_skipped=true "
                        "scale_before=%.1f scale_after=%.1f",
                        non_finite_parameter,
                        scale_before,
                        scale_after,
                    )
                else:
                    self.optimizer_steps += 1
        return float(loss.detach()), int(labels.shape[0])

    def _first_non_finite_gradient(self) -> str | None:
        for name, parameter in self.model.named_parameters():
            if (
                parameter.requires_grad
                and parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ):
                return name
        return None


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
