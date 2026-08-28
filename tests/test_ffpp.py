import json

import pytest

from deeptector.data.ffpp import (
    MANIPULATIONS,
    OfficialFFPPSplits,
    build_ffpp_manifest,
    select_smoke_records,
)


def _write_splits(directory):
    directory.mkdir()
    pairs = {
        "train.json": [["000", "003"]],
        "val.json": [["001", "870"]],
        "test.json": [["999", "960"]],
    }
    for name, payload in pairs.items():
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    return pairs


def _write_dataset(root, pairs):
    real_dir = root / "original_sequences" / "youtube" / "c23" / "videos"
    real_dir.mkdir(parents=True)
    for pair in pairs.values():
        for source in pair[0]:
            (real_dir / f"{source}.mp4").touch()
    for manipulation in MANIPULATIONS:
        directory = root / "manipulated_sequences" / manipulation / "c23" / "videos"
        directory.mkdir(parents=True)
        for pair in pairs.values():
            (directory / f"{pair[0][0]}_{pair[0][1]}.mp4").touch()


def test_ffpp_official_split_manifest_and_report(tmp_path):
    split_dir = tmp_path / "splits"
    pairs = _write_splits(split_dir)
    root = tmp_path / "ffpp"
    _write_dataset(root, pairs)
    splits = OfficialFFPPSplits.from_directory(split_dir)
    records, report = build_ffpp_manifest(root, splits)
    assert len(records) == 18
    assert report.valid
    assert report.counts_by_split_and_label["train"] == {"real": 2, "fake": 4}
    fake = next(record for record in records if record.manipulation_type == "Deepfakes")
    assert fake.source_video_id == "000|003"
    smoke = select_smoke_records(records)
    assert len(smoke) == 15


def test_ffpp_refuses_missing_official_split_files(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"Official FF\+\+ split metadata"):
        OfficialFFPPSplits.from_directory(tmp_path)


def test_ffpp_refuses_source_overlap_between_official_splits(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.json").write_text(json.dumps([["000", "003"]]), encoding="utf-8")
    (split_dir / "val.json").write_text(json.dumps([["000", "870"]]), encoding="utf-8")
    (split_dir / "test.json").write_text(json.dumps([["999", "960"]]), encoding="utf-8")
    with pytest.raises(ValueError, match="Source 000"):
        OfficialFFPPSplits.from_directory(split_dir)
