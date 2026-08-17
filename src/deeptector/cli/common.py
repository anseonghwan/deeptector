"""Shared configuration and construction helpers for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from deeptector.data.dataset import VideoFrameDataset
from deeptector.data.face_detection import OpenCVHaarFaceDetector
from deeptector.data.manifest import load_manifest
from deeptector.data.sampling import UniformFrameSampler
from deeptector.data.transforms import ImageTransform
from deeptector.models.classifier import DeepfakeClassifier
from deeptector.models.clip_encoder import CLIPVisualEncoder


def load_config(path: str | Path) -> dict[str, Any]:
    """Read an experiment YAML mapping."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Experiment config must be a YAML mapping")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Persist the effective configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def build_model(config: dict[str, Any]) -> DeepfakeClassifier:
    """Construct the configured visual baseline."""
    encoder = CLIPVisualEncoder(
        config.get("name", "openai/clip-vit-base-patch16"),
        freeze=bool(config.get("freeze_backbone", True)),
        local_files_only=bool(config.get("local_files_only", True)),
    )
    return DeepfakeClassifier(encoder, hidden_dim=config.get("hidden_dim"))


def build_loader(
    manifest_path: str | Path,
    data_config: dict[str, Any],
    *,
    split: str,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader[dict[str, object]]:
    """Build a manifest-filtered deterministic frame loader."""
    all_records = load_manifest(manifest_path)
    records = [record for record in all_records if record.split == split]
    if not records:
        raise ValueError(f"Manifest {manifest_path} contains no {split!r} records")
    if data_config.get("frame_sampling_strategy", "uniform") != "uniform":
        raise ValueError("Milestone 1 supports only uniform frame sampling")
    if int(data_config.get("clip_length", 1)) != 1:
        raise ValueError("Milestone 1 is frame-based and requires clip_length: 1")
    dataset = VideoFrameDataset(
        records,
        UniformFrameSampler(int(data_config.get("frames_per_video", 8))),
        ImageTransform(
            int(data_config.get("input_resolution", 224)),
            str(data_config.get("normalization", "clip")),
        ),
        OpenCVHaarFaceDetector(),
        face_margin=float(data_config.get("face_margin", 0.2)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
