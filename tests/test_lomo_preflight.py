import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from deeptector.benchmarking import lomo_preflight
from deeptector.benchmarking.lomo_preflight import (
    _ensure_output_under_runs,
    _run_label_free_batch,
    run_lomo_preflight,
    select_lomo_preflight_records,
)
from deeptector.data.ffpp import MANIPULATIONS
from deeptector.data.manifest import VideoRecord


def _records():
    records = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for category_index, category in enumerate(("original", *MANIPULATIONS)):
            for suffix in ("b", "a"):
                source = f"{split_index}{category_index}{suffix}"
                records.append(
                    VideoRecord(
                        video_id=f"{category}:{source}",
                        video_path=f"{source}.mp4",
                        dataset="ffpp",
                        split=split,
                        label=0 if category == "original" else 1,
                        source_video_id=source,
                        manipulation_type=category,
                        compression="c23",
                    )
                )
    return records


def test_preflight_selection_is_bounded_deterministic_and_fold_aware():
    selected = select_lomo_preflight_records(_records(), "Deepfakes")

    assert len(selected["train"]) == 4
    assert len(selected["validation"]) == 4
    assert len(selected["test"]) == 2
    assert [record.manipulation_type for record in selected["train"]] == [
        "original",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    ]
    assert [record.manipulation_type for record in selected["test"]] == [
        "original",
        "Deepfakes",
    ]
    assert all(record.video_id.endswith("a") for records in selected.values() for record in records)


def test_preflight_selection_rejects_unknown_fold():
    with pytest.raises(ValueError, match="Unsupported held-out"):
        select_lomo_preflight_records(_records(), "Unknown")


def test_preflight_output_is_restricted_to_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / "runs" / "m2a" / "preflight.json"
    assert _ensure_output_under_runs("runs/m2a/preflight.json") == expected
    with pytest.raises(ValueError, match="must be written under"):
        _ensure_output_under_runs(Path("docs") / "preflight.json")


def test_held_out_preflight_forward_does_not_consume_labels():
    class FiniteModel(torch.nn.Module):
        def forward(self, images):
            return SimpleNamespace(logits=torch.zeros(images.shape[0], device=images.device))

    class ForbiddenLabel:
        def to(self, _device):
            raise AssertionError("held-out label was consumed")

    batch = {"image": torch.ones(2, 3, 8, 8), "label": ForbiddenLabel()}
    assert (
        _run_label_free_batch(FiniteModel(), batch, device=torch.device("cpu"), amp_enabled=False)
        == 2
    )


def test_preflight_requires_xpu_and_persists_failure_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "fold.yaml"
    config_path.write_text("fixture\n", encoding="utf-8")
    config = {"experiment_name": "m2a_lomo_fixture", "seed": 42, "data": {}}
    monkeypatch.setattr(lomo_preflight, "load_config", lambda _path: config)
    monkeypatch.setattr(
        lomo_preflight,
        "validate_configured_data_protocol",
        lambda _data: {"held_out_manipulation": "Deepfakes"},
    )
    output = tmp_path / "runs" / "m2a_lomo_fixture" / "preflight.json"

    with pytest.raises(ValueError, match="requires an Intel XPU"):
        run_lomo_preflight(config_path, output, device_request="cpu")

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["success"] is False
    assert report["device"]["selected"] == "cpu"
    assert report["failure"]["type"] == "ValueError"
    assert report["checkpoint_written"] is False
