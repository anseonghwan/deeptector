import os
from pathlib import Path

import pytest

from deeptector.benchmarking.xpu_capacity import (
    GIB,
    _host_memory_snapshot,
    ensure_output_under_runs,
    estimate_full_workload,
    recommend_batch,
    summarize_repetitions,
)


@pytest.mark.skipif(os.name != "nt", reason="Windows memory API test")
def test_host_memory_snapshot_reports_windows_process_memory():
    snapshot = _host_memory_snapshot()
    assert snapshot["total_physical_bytes"] > 0
    assert snapshot["available_physical_bytes"] > 0
    assert snapshot["process_rss_bytes"] > 0
    assert snapshot["process_private_bytes"] > 0


def _repetition(e2e, compute, *, frames=32, available=16 * GIB, reserved=2 * GIB):
    return {
        "success": True,
        "device_survived": True,
        "processed_frames": frames,
        "steps": [
            {
                "end_to_end_seconds": duration,
                "compute_seconds": compute[index],
            }
            for index, duration in enumerate(e2e)
        ],
        "xpu_memory_after": {
            "max_allocated_bytes": reserved // 2,
            "max_reserved_bytes": reserved,
        },
        "available_system_memory_samples_bytes": [available],
        "process_rss_samples_bytes": [3 * GIB],
        "error": None,
    }


def _batch_result(batch, fps, *, reserved=2 * GIB):
    duration = 32 / fps
    repetitions = [_repetition([duration], [duration / 2], reserved=reserved) for _ in range(3)]
    summary = summarize_repetitions(repetitions)
    validation = dict(summary)
    validation["validation_frames_per_second"] = validation.pop("end_to_end_frames_per_second")
    return {
        "batch_size": batch,
        "xpu_total_memory_bytes": 16 * GIB,
        "training_summary": summary,
        "validation_summary": validation,
    }


def test_repetition_summary_uses_all_measured_frames():
    rows = [_repetition([1.0, 1.0], [0.5, 0.5]) for _ in range(3)]
    summary = summarize_repetitions(rows)
    assert summary["stable"]
    assert summary["total_frames"] == 96
    assert summary["end_to_end_frames_per_second"] == pytest.approx(16.0)
    assert summary["compute_frames_per_second"] == pytest.approx(32.0)


def test_recommendation_prefers_batch_four_for_low_gain_and_high_memory():
    results = [
        _batch_result(2, 10.0),
        _batch_result(4, 20.0, reserved=2 * GIB),
        _batch_result(8, 21.0, reserved=3 * GIB),
    ]
    recommendation = recommend_batch(results, 32 * GIB)
    assert recommendation["recommended_batch_size"] == 4


def test_workload_estimate_uses_frame_batch_ceiling():
    result = estimate_full_workload(8, 16.0, 20.0)
    assert result["train_steps_per_epoch"] == 3600
    assert result["validation_steps_per_epoch"] == 700
    assert result["training_seconds_per_epoch"] == 1800.0
    assert result["validation_seconds_per_epoch"] == 280.0


def test_capacity_output_must_remain_under_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ensure_output_under_runs("runs/capacity/report.json") == (
        Path.cwd() / "runs" / "capacity" / "report.json"
    )
    with pytest.raises(ValueError, match="must be written under"):
        ensure_output_under_runs("docs/report.json")
