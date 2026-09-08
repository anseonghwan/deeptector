"""Dataset-independent manifest parsing and leakage validation."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"video_id", "video_path", "dataset", "split", "label"}
VALID_SPLITS = {"train", "validation", "val", "test"}


@dataclass(frozen=True)
class VideoRecord:
    """One video and its optional forensic metadata."""

    video_id: str
    video_path: str
    dataset: str
    split: str
    label: int
    identity_id: str | None = None
    source_video_id: str | None = None
    manipulation_type: str | None = None
    manipulation_family: str | None = None
    compression: str | None = None
    fps: float | None = None
    duration: float | None = None


def _optional(value: Any) -> Any:
    return None if pd.isna(value) or value == "" else value


def load_manifest(
    path: str | Path,
    *,
    validate: bool = True,
    expand_video_paths: bool = True,
) -> list[VideoRecord]:
    """Load CSV or JSONL into typed video records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        frame = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        raise ValueError("Manifest must be .csv, .json, or .jsonl")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    names = {field.name for field in fields(VideoRecord)}
    records: list[VideoRecord] = []
    for row_number, row in frame.iterrows():
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Row {row_number}: label must be 0 (REAL) or 1 (FAKE)")
        split = str(row["split"]).lower()
        if split not in VALID_SPLITS:
            raise ValueError(f"Row {row_number}: unsupported split {split!r}")
        values = {name: _optional(row[name]) for name in names if name in frame.columns}
        video_path = str(row["video_path"])
        if expand_video_paths:
            video_path = str(Path(os.path.expandvars(video_path)))
        values.update(
            video_id=str(row["video_id"]),
            video_path=video_path,
            dataset=str(row["dataset"]),
            split="validation" if split == "val" else split,
            label=label,
        )
        records.append(VideoRecord(**values))
    if validate:
        assert_no_split_leakage(records)
    return records


def assert_no_split_leakage(
    records: Iterable[VideoRecord],
    *,
    identity_disjoint: bool = False,
    source_disjoint: bool = True,
) -> None:
    """Raise when video/source/identity groups cross data splits."""
    records = list(records)
    _assert_unique_video_ids(records)
    _assert_group_disjoint(records, "video_id")
    if source_disjoint:
        _assert_source_disjoint(records)
    if identity_disjoint:
        _assert_group_disjoint(records, "identity_id")


def _assert_group_disjoint(records: list[VideoRecord], attribute: str) -> None:
    groups: dict[str, set[str]] = {}
    for record in records:
        value = getattr(record, attribute)
        if value is not None:
            groups.setdefault(str(value), set()).add(record.split)
    leaked = {key: splits for key, splits in groups.items() if len(splits) > 1}
    if leaked:
        examples = list(leaked.items())[:5]
        raise ValueError(f"Split leakage detected for {attribute}: {examples}")


def source_ids(record: VideoRecord) -> tuple[str, ...]:
    """Return every source identity encoded in a manifest lineage field."""
    if record.source_video_id is None:
        return ()
    return tuple(part.strip() for part in str(record.source_video_id).split("|") if part.strip())


def _assert_source_disjoint(records: list[VideoRecord]) -> None:
    groups: dict[str, set[str]] = {}
    for record in records:
        for source_id in source_ids(record):
            groups.setdefault(source_id, set()).add(record.split)
    leaked = {key: splits for key, splits in groups.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Split leakage detected for source_video_id: {list(leaked.items())[:5]}")


def _assert_unique_video_ids(records: list[VideoRecord]) -> None:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.video_id] = counts.get(record.video_id, 0) + 1
    duplicates = sorted(video_id for video_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate video_id values detected: {duplicates[:5]}")


def filter_records(
    records: Iterable[VideoRecord],
    *,
    split: str | None = None,
    dataset: str | None = None,
) -> list[VideoRecord]:
    """Select records without coupling consumers to manifest storage."""
    return [
        record
        for record in records
        if (split is None or record.split == split)
        and (dataset is None or record.dataset == dataset)
    ]
