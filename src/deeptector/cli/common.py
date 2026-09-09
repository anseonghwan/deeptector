"""Shared configuration and construction helpers for CLI commands."""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from deeptector.data.dataset import VideoFrameDataset
from deeptector.data.face_detection import OpenCVHaarFaceDetector
from deeptector.data.ffpp_lomo import validate_ffpp_lomo_artifact
from deeptector.data.manifest import VideoRecord, load_manifest
from deeptector.data.sampling import UniformFrameSampler
from deeptector.data.transforms import ImageTransform
from deeptector.models.classifier import DeepfakeClassifier
from deeptector.models.clip_encoder import CLIPVisualEncoder

LOGGER = logging.getLogger(__name__)


def validate_configured_data_protocol(data_config: dict[str, Any]) -> dict[str, Any] | None:
    """Validate optional LOMO bindings before model or run initialization."""
    protocol = data_config.get("lomo_protocol")
    fold = data_config.get("lomo_fold")
    if protocol is None and fold is None:
        return None
    if not protocol or not fold:
        raise ValueError("LOMO data config requires both lomo_protocol and lomo_fold")
    manifest_keys = ("train_manifest", "validation_manifest", "test_manifest")
    missing_manifests = [key for key in manifest_keys if not data_config.get(key)]
    if missing_manifests:
        raise ValueError(
            "LOMO data config is missing required manifests: " + ", ".join(missing_manifests)
        )
    manifests = {str(data_config[key]) for key in manifest_keys}
    if len(manifests) != 1:
        raise ValueError("LOMO train, validation, and test must use the same fold manifest")
    binding = validate_ffpp_lomo_artifact(
        protocol,
        str(fold),
        manifests.pop(),
        require_frozen_m1=bool(data_config.get("lomo_require_frozen_m1", True)),
    )
    configured_held_out = data_config.get("lomo_held_out_manipulation")
    if configured_held_out != binding["held_out_manipulation"]:
        raise ValueError(
            "Configured LOMO held-out manipulation does not match protocol: "
            f"{configured_held_out!r} != {binding['held_out_manipulation']!r}"
        )
    return binding


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
    model_name = os.path.expandvars(str(config.get("name", "openai/clip-vit-base-patch16")))
    encoder = CLIPVisualEncoder(
        model_name,
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
    balanced_sampling: bool = False,
    seed: int = 42,
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
    label_counts = Counter(record.label for record in records)
    sampler = None
    if balanced_sampling:
        if split != "train":
            raise ValueError("Class-balanced sampling is permitted only for the train split")
        sampler = build_class_balanced_sampler(records, dataset.samples, seed=seed)
    LOGGER.info(
        "split=%s label_counts=%s balanced_sampling=%s strategy=%s samples_per_epoch=%d",
        split,
        dict(sorted(label_counts.items())),
        balanced_sampling,
        "inverse_manifest_label_frequency" if balanced_sampling else "none",
        len(dataset),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_class_balanced_sampler(
    records: list[VideoRecord], samples: list[tuple[int, int]], *, seed: int
) -> WeightedRandomSampler:
    """Weight train frames by inverse manifest-level label frequency."""
    labels = [int(record.label) for record in records]
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("Balanced sampling requires at least two training labels")
    class_weights = {label: 1.0 / count for label, count in counts.items()}
    sample_weights = [class_weights[labels[record_index]] for record_index, _ in samples]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
