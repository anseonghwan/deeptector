"""Deterministic FaceForensics++ leave-one-manipulation-out protocols."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .ffpp import MANIPULATIONS, PAIR_PATTERN, SOURCE_PATTERN, write_manifest
from .manifest import VideoRecord, assert_no_split_leakage, load_manifest, source_ids

PROTOCOL_VERSION = "ffpp-c23-lomo-v1"
FROZEN_M1_MANIFEST_SHA256 = "ebee8dbfd4ce9dd23228a2063a8adf22dfe2e4c740a19a12ba6f7ea63eea2ab2"
FOLD_SLUGS = {
    "Deepfakes": "deepfakes",
    "Face2Face": "face2face",
    "FaceSwap": "faceswap",
    "NeuralTextures": "neuraltextures",
}
SPLITS = ("train", "validation", "test")
SPLIT_ORDER = {split: index for index, split in enumerate(SPLITS)}
FROZEN_M1_EXPECTED_FOLD_COUNTS = {
    "train": {"real": 720, "fake": 2160, "total": 2880},
    "validation": {"real": 140, "fake": 420, "total": 560},
    "test": {"real": 140, "fake": 140, "total": 280},
}


@dataclass(frozen=True)
class FFPPLOMOFold:
    """One validated LOMO fold and its auditable metadata."""

    held_out_manipulation: str
    slug: str
    records: tuple[VideoRecord, ...]
    metadata: dict[str, Any]


def build_ffpp_lomo_folds(records: list[VideoRecord]) -> dict[str, FFPPLOMOFold]:
    """Build all four deterministic folds from official FF++ assignments."""
    _validate_source_records(records)
    return {
        FOLD_SLUGS[manipulation]: _build_validated_fold(records, manipulation)
        for manipulation in MANIPULATIONS
    }


def build_ffpp_lomo_fold(records: list[VideoRecord], held_out_manipulation: str) -> FFPPLOMOFold:
    """Build one fold while preserving every selected record's official split."""
    if held_out_manipulation not in MANIPULATIONS:
        raise ValueError(
            f"Unsupported held-out manipulation {held_out_manipulation!r}; "
            f"expected one of {MANIPULATIONS}"
        )
    _validate_source_records(records)
    return _build_validated_fold(records, held_out_manipulation)


def generate_ffpp_lomo_protocol(
    source_manifest: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Validate, generate, and atomically replace each LOMO output file."""
    source_path = Path(source_manifest)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source manifest does not exist: {source_path}")
    source_sha256 = _sha256(source_path)
    records = load_manifest(source_path, expand_video_paths=False)
    _validate_source_records(records)
    missing_files = _missing_video_paths(records)
    if missing_files:
        raise ValueError(
            "LOMO source manifest references missing video files: "
            + json.dumps(missing_files[:10], indent=2)
        )
    folds = build_ffpp_lomo_folds(records)
    output_path = Path(output_directory)
    targets = {slug: output_path / f"{slug}.csv" for slug in FOLD_SLUGS.values()}
    protocol_path = output_path / "protocol.json"
    _assert_source_not_overwritten(source_path, [*targets.values(), protocol_path])

    source_matches_frozen_m1 = source_sha256 == FROZEN_M1_MANIFEST_SHA256
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "deeptector_ffpp_lomo_protocol",
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "held-out FaceForensics++ manipulation transfer",
        "cross_dataset": False,
        "identity_disjoint": False,
        "universal_generalization_claim": False,
        "source_manifest": {
            "path": source_path.as_posix(),
            "sha256": source_sha256,
            "records": len(records),
            "matches_frozen_m1_manifest": source_matches_frozen_m1,
        },
        "fold_order": [FOLD_SLUGS[value] for value in MANIPULATIONS],
        "integrity": {
            "duplicate_video_ids": 0,
            "missing_files": 0,
            "unknown_manipulation_values": 0,
            "source_lineage_overlap_across_splits": 0,
            "source_split_assignments_preserved": True,
            "official_split_verification": {
                "verified": source_matches_frozen_m1,
                "basis": "frozen_m1_manifest_sha256" if source_matches_frozen_m1 else None,
            },
        },
        "folds": {},
    }

    output_path.mkdir(parents=True, exist_ok=True)
    temporary_paths: dict[str, Path] = {}
    try:
        for slug, fold in folds.items():
            target = targets[slug]
            temporary = target.with_suffix(target.suffix + ".tmp")
            _assert_source_not_overwritten(source_path, [temporary])
            write_manifest(list(fold.records), temporary)
            temporary_paths[slug] = temporary
            protocol["folds"][slug] = {
                **fold.metadata,
                "manifest": target.name,
                "manifest_sha256": _sha256(temporary),
            }
        if source_matches_frozen_m1:
            _assert_frozen_m1_counts(protocol["folds"])
        protocol["frozen_m1_counts_asserted"] = source_matches_frozen_m1
        if _sha256(source_path) != source_sha256:
            raise RuntimeError("Source manifest changed during LOMO protocol generation")
        protocol_temporary = protocol_path.with_suffix(protocol_path.suffix + ".tmp")
        _assert_source_not_overwritten(source_path, [protocol_temporary])
        protocol_temporary.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for slug, temporary in temporary_paths.items():
            temporary.replace(targets[slug])
        protocol_temporary.replace(protocol_path)
    finally:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        protocol_path.with_suffix(protocol_path.suffix + ".tmp").unlink(missing_ok=True)
    return protocol


def _build_validated_fold(records: list[VideoRecord], held_out_manipulation: str) -> FFPPLOMOFold:
    selected = []
    for record in records:
        if record.split in {"train", "validation"}:
            if record.label == 0 or record.manipulation_type != held_out_manipulation:
                selected.append(record)
        elif record.split == "test" and (
            record.label == 0 or record.manipulation_type == held_out_manipulation
        ):
            selected.append(record)
    selected.sort(key=lambda record: (SPLIT_ORDER[record.split], record.video_id))
    assert_no_split_leakage(selected)
    metadata = _fold_metadata(records, selected, held_out_manipulation)
    return FFPPLOMOFold(
        held_out_manipulation=held_out_manipulation,
        slug=FOLD_SLUGS[held_out_manipulation],
        records=tuple(selected),
        metadata=metadata,
    )


def _validate_source_records(records: list[VideoRecord]) -> None:
    if not records:
        raise ValueError("LOMO source manifest must not be empty")
    assert_no_split_leakage(records)
    invalid_splits = sorted({record.split for record in records if record.split not in SPLITS})
    if invalid_splits:
        raise ValueError(f"LOMO records have invalid splits: {invalid_splits}")
    invalid_labels = sorted({record.label for record in records if record.label not in {0, 1}})
    if invalid_labels:
        raise ValueError(f"LOMO records have invalid labels: {invalid_labels}")
    invalid_datasets = sorted({record.dataset for record in records if record.dataset != "ffpp"})
    if invalid_datasets:
        raise ValueError(f"LOMO supports only dataset='ffpp'; found: {invalid_datasets}")
    invalid_compressions = sorted(
        {str(record.compression) for record in records if record.compression != "c23"}
    )
    if invalid_compressions:
        raise ValueError(f"LOMO protocol requires FF++ c23 records; found: {invalid_compressions}")
    unknown = sorted(
        {
            str(record.manipulation_type)
            for record in records
            if (record.label == 0 and record.manipulation_type != "original")
            or (record.label == 1 and record.manipulation_type not in MANIPULATIONS)
        }
    )
    if unknown:
        raise ValueError(f"Unknown or inconsistent FF++ manipulation values: {unknown}")
    for record in records:
        _validate_canonical_ffpp_record(record)
    missing_lineage = sorted(record.video_id for record in records if not source_ids(record))
    if missing_lineage:
        raise ValueError(f"LOMO records have no source lineage: {missing_lineage[:5]}")
    counts = _counts_by_split_and_manipulation(records)
    for split in SPLITS:
        if counts[split].get("original", 0) == 0:
            raise ValueError(f"LOMO source manifest has no real records in {split}")
        missing = [value for value in MANIPULATIONS if counts[split].get(value, 0) == 0]
        if missing:
            raise ValueError(f"LOMO source manifest is missing {missing} in {split}")


def _fold_metadata(
    source_records: list[VideoRecord],
    fold_records: list[VideoRecord],
    held_out_manipulation: str,
) -> dict[str, Any]:
    counts_by_split = _counts_by_split(fold_records)
    manipulation_counts = _counts_by_split_and_manipulation(fold_records)
    held_out_counts = {
        split: manipulation_counts[split].get(held_out_manipulation, 0) for split in SPLITS
    }
    if held_out_counts["train"] or held_out_counts["validation"]:
        raise ValueError(f"Held-out manipulation entered train/validation: {held_out_counts}")
    test_fake_manipulations = sorted(
        {
            str(record.manipulation_type)
            for record in fold_records
            if record.split == "test" and record.label == 1
        }
    )
    if test_fake_manipulations != [held_out_manipulation]:
        raise ValueError(
            "Held-out test must contain exactly the target fake manipulation; found: "
            f"{test_fake_manipulations}"
        )
    source_assignments = {record.video_id: record.split for record in source_records}
    source_assignments_preserved = all(
        source_assignments.get(record.video_id) == record.split for record in fold_records
    )
    if not source_assignments_preserved:
        raise ValueError("A LOMO record no longer has its source manifest split assignment")
    overlaps = _source_lineage_overlap(fold_records)
    if overlaps:
        raise ValueError(f"LOMO source lineage crosses splits: {overlaps}")
    duplicate_count = len(fold_records) - len({record.video_id for record in fold_records})
    if duplicate_count:
        raise ValueError(f"LOMO fold contains {duplicate_count} duplicate video IDs")
    return {
        "held_out_manipulation": held_out_manipulation,
        "counts_by_split": {
            split: len([record for record in fold_records if record.split == split])
            for split in SPLITS
        },
        "counts_by_split_and_label": counts_by_split,
        "counts_by_split_and_manipulation": manipulation_counts,
        "held_out_manipulation_counts": held_out_counts,
        "held_out_test_fake_manipulations": test_fake_manipulations,
        "duplicate_video_ids": duplicate_count,
        "source_lineage_overlap_across_splits": overlaps,
        "source_split_assignments_preserved": source_assignments_preserved,
    }


def _validate_canonical_ffpp_record(record: VideoRecord) -> None:
    parts = record.video_id.split(":")
    if len(parts) != 4:
        raise ValueError(f"Non-canonical FF++ video_id: {record.video_id}")
    dataset, manipulation, compression, stem = parts
    if dataset != "ffpp" or compression != "c23":
        raise ValueError(f"Non-canonical FF++ video_id: {record.video_id}")
    if manipulation != record.manipulation_type:
        raise ValueError(
            "FF++ manipulation metadata mismatch for "
            f"{record.video_id}: video_id={manipulation!r}, field={record.manipulation_type!r}"
        )

    portable_path = record.video_path.replace("\\", "/")
    if PurePosixPath(portable_path).stem != stem:
        raise ValueError(f"FF++ video_id/path stem mismatch for {record.video_id}")
    lineage = source_ids(record)
    if record.label == 0:
        expected_directory = "/original_sequences/youtube/c23/videos/"
        if not SOURCE_PATTERN.fullmatch(stem) or lineage != (stem,):
            raise ValueError(f"Invalid FF++ real lineage for {record.video_id}: {lineage}")
    else:
        expected_directory = f"/manipulated_sequences/{manipulation}/c23/videos/"
        match = PAIR_PATTERN.fullmatch(stem)
        if match is None or lineage != match.groups():
            raise ValueError(f"Invalid FF++ fake lineage for {record.video_id}: {lineage}")
    if expected_directory not in portable_path:
        raise ValueError(
            f"FF++ manipulation path mismatch for {record.video_id}: {record.video_path}"
        )


def _counts_by_split(records: list[VideoRecord]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_records = [record for record in records if record.split == split]
        real = sum(record.label == 0 for record in split_records)
        fake = sum(record.label == 1 for record in split_records)
        result[split] = {"real": real, "fake": fake, "total": len(split_records)}
    return result


def _counts_by_split_and_manipulation(
    records: list[VideoRecord],
) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for record in records:
        counters[record.split][str(record.manipulation_type)] += 1
    return {split: dict(sorted(counters[split].items())) for split in SPLITS}


def _source_lineage_overlap(records: list[VideoRecord]) -> dict[str, list[str]]:
    assignments: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for source in source_ids(record):
            assignments[source].add(record.split)
    return {
        source: sorted(splits) for source, splits in sorted(assignments.items()) if len(splits) > 1
    }


def _missing_video_paths(records: list[VideoRecord]) -> list[str]:
    missing = []
    for record in records:
        expanded = Path(os.path.expandvars(record.video_path))
        if not expanded.is_file():
            missing.append(record.video_path)
    return sorted(missing)


def _assert_source_not_overwritten(source: Path, targets: list[Path]) -> None:
    source_resolved = source.resolve()
    for target in targets:
        if target.resolve() == source_resolved:
            raise ValueError(f"LOMO output cannot overwrite its source manifest: {source}")


def _assert_frozen_m1_counts(folds: dict[str, dict[str, Any]]) -> None:
    for slug, metadata in folds.items():
        if metadata["counts_by_split_and_label"] != FROZEN_M1_EXPECTED_FOLD_COUNTS:
            raise ValueError(
                f"Frozen M1 manifest produced unexpected counts for fold {slug}: "
                f"{metadata['counts_by_split_and_label']}"
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
