from pathlib import Path

import pytest

from deeptector.cli import common
from deeptector.cli.common import load_config, validate_configured_data_protocol
from deeptector.cli.train import validate_lomo_run_state

CONFIG_DIRECTORY = Path("configs/experiment/m2a_lomo")
FOLDS = {
    "deepfakes": "Deepfakes",
    "face2face": "Face2Face",
    "faceswap": "FaceSwap",
    "neuraltextures": "NeuralTextures",
}


def test_all_lomo_configs_preserve_m1_training_definition():
    m1 = load_config("configs/experiment/ffpp_m1.yaml")
    for slug, held_out in FOLDS.items():
        config = load_config(CONFIG_DIRECTORY / f"{slug}_seed42.yaml")
        assert config["experiment_name"] == f"m2a_lomo_{slug}_seed42"
        assert config["seed"] == m1["seed"] == 42
        assert config["model"] == m1["model"]
        assert config["train"] == m1["train"]
        assert config["evaluation"] == m1["evaluation"]
        assert config["data"]["lomo_fold"] == slug
        assert config["data"]["lomo_held_out_manipulation"] == held_out
        assert config["data"]["lomo_require_frozen_m1"] is True
        manifest = f"data/manifests/ffpp_lomo_c23/{slug}.csv"
        assert config["data"]["train_manifest"] == manifest
        assert config["data"]["validation_manifest"] == manifest
        assert config["data"]["test_manifest"] == manifest
        for key in (
            "compression",
            "frames_per_video",
            "frame_sampling_strategy",
            "clip_length",
            "input_resolution",
            "face_margin",
            "normalization",
            "identity_disjoint",
        ):
            assert config["data"][key] == m1["data"][key]


def test_lomo_config_requires_one_shared_fold_manifest(monkeypatch):
    monkeypatch.setattr(common, "validate_ffpp_lomo_artifact", lambda *args, **kwargs: {})
    data = {
        "lomo_protocol": "protocol.json",
        "lomo_fold": "deepfakes",
        "lomo_held_out_manipulation": "Deepfakes",
        "train_manifest": "deepfakes.csv",
        "validation_manifest": "other.csv",
        "test_manifest": "deepfakes.csv",
    }
    with pytest.raises(ValueError, match="same fold manifest"):
        validate_configured_data_protocol(data)


def test_lomo_config_requires_all_three_manifests(monkeypatch):
    monkeypatch.setattr(common, "validate_ffpp_lomo_artifact", lambda *args, **kwargs: {})
    data = {
        "lomo_protocol": "protocol.json",
        "lomo_fold": "deepfakes",
        "lomo_held_out_manipulation": "Deepfakes",
        "train_manifest": "deepfakes.csv",
        "validation_manifest": "deepfakes.csv",
    }
    with pytest.raises(ValueError, match="missing required manifests: test_manifest"):
        validate_configured_data_protocol(data)


def test_lomo_config_requires_matching_held_out_metadata(monkeypatch):
    monkeypatch.setattr(
        common,
        "validate_ffpp_lomo_artifact",
        lambda *args, **kwargs: {"held_out_manipulation": "Deepfakes"},
    )
    data = {
        "lomo_protocol": "protocol.json",
        "lomo_fold": "deepfakes",
        "lomo_held_out_manipulation": "FaceSwap",
        "train_manifest": "deepfakes.csv",
        "validation_manifest": "deepfakes.csv",
        "test_manifest": "deepfakes.csv",
    }
    with pytest.raises(ValueError, match="held-out manipulation"):
        validate_configured_data_protocol(data)


def test_non_lomo_config_remains_unchanged():
    assert (
        validate_configured_data_protocol(load_config("configs/experiment/ffpp_m1.yaml")["data"])
        is None
    )


def test_lomo_fresh_training_rejects_existing_checkpoint_state(tmp_path):
    run_dir = tmp_path / "runs" / "fold"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.pt").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="explicit --resume-from"):
        validate_lomo_run_state(run_dir, resume_from=None, config={"fold": "deepfakes"})


def test_lomo_resume_requires_own_last_and_best_checkpoints(tmp_path):
    run_dir = tmp_path / "runs" / "fold"
    run_dir.mkdir(parents=True)
    (run_dir / "last_checkpoint.pt").write_bytes(b"last")

    with pytest.raises(ValueError, match="must use last_checkpoint.pt"):
        validate_lomo_run_state(
            run_dir,
            resume_from=run_dir / "checkpoint.pt",
            config={"fold": "deepfakes"},
        )
    with pytest.raises(FileNotFoundError, match="best checkpoint is missing"):
        validate_lomo_run_state(
            run_dir,
            resume_from=run_dir / "last_checkpoint.pt",
            config={"fold": "deepfakes"},
        )

    (run_dir / "checkpoint.pt").write_bytes(b"best")
    validate_lomo_run_state(
        run_dir,
        resume_from=run_dir / "last_checkpoint.pt",
        config={"fold": "deepfakes"},
    )
