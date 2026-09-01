"""FaceForensics++ discovery using its official split metadata."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .manifest import VideoRecord, assert_no_split_leakage

MANIPULATIONS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
PAIR_PATTERN = re.compile(r"^(\d{3})_(\d{3})$")
SOURCE_PATTERN = re.compile(r"^\d{3}$")


@dataclass(frozen=True)
class OfficialFFPPSplits:
    """Official FF++ pair and source assignments."""

    pair_to_split: dict[str, str]
    source_to_split: dict[str, str]

    @classmethod
    def from_directory(cls, directory: str | Path) -> OfficialFFPPSplits:
        """Load official train.json, val.json, and test.json pair lists."""
        directory = Path(directory)
        files = {
            "train": directory / "train.json",
            "validation": directory / "val.json",
            "test": directory / "test.json",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Official FF++ split metadata is required; missing: " + ", ".join(missing)
            )
        pair_to_split: dict[str, str] = {}
        source_to_split: dict[str, str] = {}
        for split, path in files.items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise TypeError(f"Official split must be a JSON list: {path}")
            for index, pair in enumerate(payload):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(f"{path}:{index} must contain a two-source list")
                first, second = (str(value).zfill(3) for value in pair)
                if not SOURCE_PATTERN.fullmatch(first) or not SOURCE_PATTERN.fullmatch(second):
                    raise ValueError(f"{path}:{index} has invalid source IDs: {pair}")
                key = f"{first}_{second}"
                reverse = f"{second}_{first}"
                for candidate in (key, reverse):
                    previous = pair_to_split.setdefault(candidate, split)
                    if previous != split:
                        raise ValueError(f"Pair {candidate} occurs in {previous} and {split}")
                for source in (first, second):
                    previous = source_to_split.setdefault(source, split)
                    if previous != split:
                        raise ValueError(f"Source {source} occurs in {previous} and {split}")
        return cls(pair_to_split=pair_to_split, source_to_split=source_to_split)


@dataclass
class FFPPSplitReport:
    """Auditable FF++ discovery and leakage summary."""

    counts_by_split_and_label: dict[str, dict[str, int]]
    counts_by_split_and_manipulation: dict[str, dict[str, int]]
    duplicate_video_ids: list[str]
    source_overlap_across_splits: dict[str, list[str]]
    missing_files: list[str]
    unknown_files: list[str]
    total_records: int

    @property
    def valid(self) -> bool:
        return not any(
            (
                self.duplicate_video_ids,
                self.source_overlap_across_splits,
                self.missing_files,
                self.unknown_files,
            )
        )


def resolve_ffpp_root(explicit: str | Path | None = None) -> Path:
    """Resolve FF++ from an explicit path or DEEPTECTOR_DATA_ROOT."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    data_root = os.environ.get("DEEPTECTOR_DATA_ROOT")
    if not data_root:
        raise OSError("Set DEEPTECTOR_DATA_ROOT or pass --ffpp-root")
    return (Path(data_root).expanduser() / "ffpp").resolve()


def build_ffpp_manifest(
    ffpp_root: str | Path,
    splits: OfficialFFPPSplits,
    *,
    compression: str = "c23",
    portable_root: str = "${DEEPTECTOR_DATA_ROOT}/ffpp",
) -> tuple[list[VideoRecord], FFPPSplitReport]:
    """Discover FF++ videos without extracting frames or changing source data."""
    root = Path(ffpp_root)
    records: list[VideoRecord] = []
    unknown: list[str] = []
    missing: list[str] = []

    real_dir = root / "original_sequences" / "youtube" / compression / "videos"
    real_files = {path.stem: path for path in real_dir.glob("*.mp4")}
    for source, split in sorted(splits.source_to_split.items()):
        path = real_files.get(source)
        if path is None:
            missing.append(str(real_dir / f"{source}.mp4"))
            continue
        records.append(
            VideoRecord(
                video_id=f"ffpp:original:{compression}:{source}",
                video_path=(
                    f"{portable_root}/original_sequences/youtube/{compression}/videos/{path.name}"
                ),
                dataset="ffpp",
                split=split,
                label=0,
                source_video_id=source,
                manipulation_type="original",
                manipulation_family="real",
                compression=compression,
            )
        )
    unknown.extend(
        str(path) for stem, path in real_files.items() if stem not in splits.source_to_split
    )

    expected_directed_pairs = set(splits.pair_to_split)
    for manipulation in MANIPULATIONS:
        directory = root / "manipulated_sequences" / manipulation / compression / "videos"
        files = {path.stem: path for path in directory.glob("*.mp4")}
        for stem, path in sorted(files.items()):
            match = PAIR_PATTERN.fullmatch(stem)
            split = splits.pair_to_split.get(stem)
            if match is None or split is None:
                unknown.append(str(path))
                continue
            first, second = match.groups()
            records.append(
                VideoRecord(
                    video_id=f"ffpp:{manipulation}:{compression}:{stem}",
                    video_path=(
                        f"{portable_root}/manipulated_sequences/{manipulation}/"
                        f"{compression}/videos/{path.name}"
                    ),
                    dataset="ffpp",
                    split=split,
                    label=1,
                    source_video_id=f"{first}|{second}",
                    manipulation_type=manipulation,
                    manipulation_family="face_manipulation",
                    compression=compression,
                )
            )
        for pair in sorted(expected_directed_pairs - files.keys()):
            missing.append(str(directory / f"{pair}.mp4"))

    duplicate_counts = Counter(record.video_id for record in records)
    duplicates = sorted(key for key, count in duplicate_counts.items() if count > 1)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.source_video_id:
            for source in record.source_video_id.split("|"):
                source_splits[source].add(record.split)
    overlaps = {
        source: sorted(values) for source, values in source_splits.items() if len(values) > 1
    }
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    by_manipulation: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_label[record.split]["real" if record.label == 0 else "fake"] += 1
        by_manipulation[record.split][record.manipulation_type or "unknown"] += 1
    report = FFPPSplitReport(
        counts_by_split_and_label={key: dict(value) for key, value in by_label.items()},
        counts_by_split_and_manipulation={
            key: dict(value) for key, value in by_manipulation.items()
        },
        duplicate_video_ids=duplicates,
        source_overlap_across_splits=overlaps,
        missing_files=missing,
        unknown_files=unknown,
        total_records=len(records),
    )
    if not report.valid:
        raise ValueError("FF++ split validation failed:\n" + json.dumps(asdict(report), indent=2))
    assert_no_split_leakage(records)
    return records, report


def write_manifest(records: list[VideoRecord], path: str | Path) -> None:
    """Write one combined manifest while preserving portable video paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0])) if records else list(VideoRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def write_split_report(report: FFPPSplitReport, path: str | Path) -> None:
    """Persist the pre-training split audit as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def select_smoke_records(
    records: list[VideoRecord], *, per_category_per_split: int = 1
) -> list[VideoRecord]:
    """Select a bounded official-split subset with real and every manipulation."""
    if per_category_per_split < 1:
        raise ValueError("per_category_per_split must be positive")
    selected: list[VideoRecord] = []
    for split in ("train", "validation", "test"):
        for category in ("original", *MANIPULATIONS):
            matches = sorted(
                (
                    record
                    for record in records
                    if record.split == split and record.manipulation_type == category
                ),
                key=lambda record: record.video_id,
            )
            if len(matches) < per_category_per_split:
                raise ValueError(f"Not enough {category} records in official {split} split")
            selected.extend(matches[:per_category_per_split])
    assert_no_split_leakage(selected)
    return selected
