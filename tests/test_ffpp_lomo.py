import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from deeptector.data import ffpp_lomo
from deeptector.data.ffpp import MANIPULATIONS, write_manifest
from deeptector.data.ffpp_lomo import (
    FOLD_SLUGS,
    build_ffpp_lomo_fold,
    generate_ffpp_lomo_protocol,
)
from deeptector.data.manifest import VideoRecord, assert_no_split_leakage, load_manifest


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_manifest(tmp_path, monkeypatch):
    data_root = tmp_path / "dataset-root"
    monkeypatch.setenv("DEEPTECTOR_DATA_ROOT", str(data_root))
    records = []
    sources_by_split = {
        "train": ("000", "001"),
        "validation": ("100", "101"),
        "test": ("200", "201"),
    }
    for split, sources in sources_by_split.items():
        for source in sources:
            relative_path = f"ffpp/original_sequences/youtube/c23/videos/{source}.mp4"
            path = data_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            records.append(
                VideoRecord(
                    video_id=f"ffpp:original:c23:{source}",
                    video_path=f"${{DEEPTECTOR_DATA_ROOT}}/{relative_path}",
                    dataset="ffpp",
                    split=split,
                    label=0,
                    source_video_id=source,
                    manipulation_type="original",
                    manipulation_family="pristine",
                    compression="c23",
                )
            )
        for manipulation in MANIPULATIONS:
            for first, second in (sources, tuple(reversed(sources))):
                stem = f"{first}_{second}"
                relative_path = f"ffpp/manipulated_sequences/{manipulation}/c23/videos/{stem}.mp4"
                path = data_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                records.append(
                    VideoRecord(
                        video_id=f"ffpp:{manipulation}:c23:{stem}",
                        video_path=f"${{DEEPTECTOR_DATA_ROOT}}/{relative_path}",
                        dataset="ffpp",
                        split=split,
                        label=1,
                        source_video_id=f"{first}|{second}",
                        manipulation_type=manipulation,
                        manipulation_family="face_manipulation",
                        compression="c23",
                    )
                )
    manifest = tmp_path / "ffpp_c23.csv"
    write_manifest(records, manifest)
    return manifest, records


def test_generates_all_four_lomo_folds_with_expected_composition(tmp_path, monkeypatch):
    source, source_records = _write_source_manifest(tmp_path, monkeypatch)
    output = tmp_path / "lomo"

    protocol = generate_ffpp_lomo_protocol(source, output)

    assert protocol["fold_order"] == [FOLD_SLUGS[value] for value in MANIPULATIONS]
    assert set(protocol["folds"]) == set(FOLD_SLUGS.values())
    assert protocol["source_manifest"]["sha256"] == _sha256(source)
    assert protocol["source_manifest"]["matches_frozen_m1_manifest"] is False
    assert protocol["frozen_m1_counts_asserted"] is False
    assert protocol["cross_dataset"] is False
    assert protocol["identity_disjoint"] is False
    assert protocol["universal_generalization_claim"] is False
    assert protocol["integrity"]["source_split_assignments_preserved"] is True
    assert protocol["integrity"]["official_split_verification"] == {
        "verified": False,
        "basis": None,
    }

    source_assignments = {record.video_id: record.split for record in source_records}
    for manipulation, slug in FOLD_SLUGS.items():
        fold_path = output / f"{slug}.csv"
        records = load_manifest(fold_path, expand_video_paths=False)
        assert_no_split_leakage(records)
        assert all(source_assignments[record.video_id] == record.split for record in records)

        train = [record for record in records if record.split == "train"]
        validation = [record for record in records if record.split == "validation"]
        test = [record for record in records if record.split == "test"]
        assert len(train) == 8
        assert len(validation) == 8
        assert len(test) == 4
        assert not any(record.manipulation_type == manipulation for record in train)
        assert not any(record.manipulation_type == manipulation for record in validation)
        assert {record.manipulation_type for record in test if record.label == 1} == {manipulation}
        assert all(record.label == 0 or record.manipulation_type == manipulation for record in test)

        metadata = protocol["folds"][slug]
        assert metadata["counts_by_split_and_label"] == {
            "train": {"real": 2, "fake": 6, "total": 8},
            "validation": {"real": 2, "fake": 6, "total": 8},
            "test": {"real": 2, "fake": 2, "total": 4},
        }
        assert metadata["held_out_manipulation_counts"] == {
            "train": 0,
            "validation": 0,
            "test": 2,
        }
        assert metadata["source_lineage_overlap_across_splits"] == {}
        assert metadata["source_split_assignments_preserved"] is True


def test_regeneration_is_byte_deterministic_and_preserves_portable_paths(tmp_path, monkeypatch):
    source, _ = _write_source_manifest(tmp_path, monkeypatch)
    output = tmp_path / "lomo"
    generate_ffpp_lomo_protocol(source, output)
    paths = [output / f"{slug}.csv" for slug in FOLD_SLUGS.values()]
    paths.append(output / "protocol.json")
    first = {path.name: path.read_bytes() for path in paths}

    generate_ffpp_lomo_protocol(source, output)

    assert {path.name: path.read_bytes() for path in paths} == first
    assert b"${DEEPTECTOR_DATA_ROOT}" in (output / "deepfakes.csv").read_bytes()


def test_invalid_held_out_manipulation_is_rejected(tmp_path, monkeypatch):
    _, records = _write_source_manifest(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Unsupported held-out manipulation"):
        build_ffpp_lomo_fold(records, "UnknownMethod")


def test_unknown_source_manipulation_is_rejected(tmp_path, monkeypatch):
    _, records = _write_source_manifest(tmp_path, monkeypatch)
    fake_index = next(index for index, record in enumerate(records) if record.label == 1)
    records[fake_index] = replace(records[fake_index], manipulation_type="UnknownMethod")
    with pytest.raises(ValueError, match="Unknown or inconsistent"):
        build_ffpp_lomo_fold(records, "Deepfakes")


def test_known_manipulation_metadata_mismatch_is_rejected(tmp_path, monkeypatch):
    _, records = _write_source_manifest(tmp_path, monkeypatch)
    fake_index = next(index for index, record in enumerate(records) if record.label == 1)
    records[fake_index] = replace(records[fake_index], manipulation_type="Face2Face")
    with pytest.raises(ValueError, match="manipulation metadata mismatch"):
        build_ffpp_lomo_fold(records, "Deepfakes")


def test_source_lineage_leakage_is_rejected(tmp_path, monkeypatch):
    _, records = _write_source_manifest(tmp_path, monkeypatch)
    test_index = next(index for index, record in enumerate(records) if record.split == "test")
    records[test_index] = replace(records[test_index], source_video_id="000")
    with pytest.raises(ValueError, match="source_video_id"):
        build_ffpp_lomo_fold(records, "Deepfakes")


def test_missing_video_is_rejected(tmp_path, monkeypatch):
    source, records = _write_source_manifest(tmp_path, monkeypatch)
    missing = records[0].video_path.replace(
        "${DEEPTECTOR_DATA_ROOT}", str(tmp_path / "dataset-root")
    )
    Path(missing).unlink()
    with pytest.raises(ValueError, match="missing video files"):
        generate_ffpp_lomo_protocol(source, tmp_path / "lomo")


def test_generation_cannot_overwrite_source_manifest(tmp_path, monkeypatch):
    source, _ = _write_source_manifest(tmp_path, monkeypatch)
    collision = tmp_path / "deepfakes.csv"
    source.replace(collision)
    before = collision.read_bytes()

    with pytest.raises(ValueError, match="cannot overwrite"):
        generate_ffpp_lomo_protocol(collision, tmp_path)

    assert collision.read_bytes() == before


def test_generation_does_not_modify_source_manifest(tmp_path, monkeypatch):
    source, _ = _write_source_manifest(tmp_path, monkeypatch)
    before = source.read_bytes()

    generate_ffpp_lomo_protocol(source, tmp_path / "lomo")

    assert source.read_bytes() == before


def test_frozen_source_hash_enforces_frozen_counts(tmp_path, monkeypatch):
    source, _ = _write_source_manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(ffpp_lomo, "FROZEN_M1_MANIFEST_SHA256", _sha256(source))

    with pytest.raises(ValueError, match="unexpected counts"):
        generate_ffpp_lomo_protocol(source, tmp_path / "lomo")


def test_source_change_during_generation_prevents_publication(tmp_path, monkeypatch):
    source, _ = _write_source_manifest(tmp_path, monkeypatch)
    original_write_manifest = ffpp_lomo.write_manifest
    calls = 0

    def write_and_change_source(records, path):
        nonlocal calls
        original_write_manifest(records, path)
        calls += 1
        if calls == 1:
            source.write_bytes(source.read_bytes() + b"\n")

    monkeypatch.setattr(ffpp_lomo, "write_manifest", write_and_change_source)
    output = tmp_path / "lomo"

    with pytest.raises(RuntimeError, match="changed during"):
        generate_ffpp_lomo_protocol(source, output)

    assert not any((output / f"{slug}.csv").exists() for slug in FOLD_SLUGS.values())
    assert not (output / "protocol.json").exists()
