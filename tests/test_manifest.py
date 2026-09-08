import pandas as pd
import pytest

from deeptector.data.manifest import VideoRecord, assert_no_split_leakage, load_manifest


def test_manifest_parses_missing_optional_metadata(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "video_id": "v1",
                "video_path": "v1.mp4",
                "dataset": "synthetic",
                "split": "train",
                "label": 0,
            }
        ]
    ).to_csv(path, index=False)
    record = load_manifest(path)[0]
    assert record.label == 0
    assert record.identity_id is None


def test_video_leakage_is_rejected():
    records = [
        VideoRecord("v1", "a", "d", "train", 0),
        VideoRecord("v1", "a", "d", "test", 0),
    ]
    with pytest.raises(ValueError, match="video_id"):
        assert_no_split_leakage(records)


def test_identity_leakage_is_optional_and_detectable():
    records = [
        VideoRecord("v1", "a", "d", "train", 0, identity_id="person"),
        VideoRecord("v2", "b", "d", "test", 1, identity_id="person"),
    ]
    assert_no_split_leakage(records)
    with pytest.raises(ValueError, match="identity_id"):
        assert_no_split_leakage(records, identity_disjoint=True)


def test_each_source_in_composite_lineage_is_leakage_checked():
    records = [
        VideoRecord("fake-a", "a", "d", "train", 1, source_video_id="000|003"),
        VideoRecord("fake-b", "b", "d", "test", 1, source_video_id="000|870"),
    ]
    with pytest.raises(ValueError, match="source_video_id"):
        assert_no_split_leakage(records)


def test_manifest_expands_environment_video_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPTECTOR_DATA_ROOT", str(tmp_path))
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "video_id": "v1",
                "video_path": "${DEEPTECTOR_DATA_ROOT}/v1.mp4",
                "dataset": "synthetic",
                "split": "train",
                "label": 0,
            }
        ]
    ).to_csv(path, index=False)
    assert load_manifest(path)[0].video_path == str(tmp_path / "v1.mp4")


def test_manifest_can_preserve_portable_environment_video_path(tmp_path):
    path = tmp_path / "manifest.csv"
    portable_path = "${DEEPTECTOR_DATA_ROOT}/ffpp/v1.mp4"
    pd.DataFrame(
        [
            {
                "video_id": "v1",
                "video_path": portable_path,
                "dataset": "synthetic",
                "split": "train",
                "label": 0,
            }
        ]
    ).to_csv(path, index=False)

    assert load_manifest(path, expand_video_paths=False)[0].video_path == portable_path
