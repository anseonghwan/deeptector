"""Run bounded FF++ decode, face, encoder, and determinism validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deeptector.cli.common import build_model, load_config
from deeptector.data.dataset import OpenCVVideoReader
from deeptector.data.face_detection import OpenCVHaarFaceDetector, center_square_crop
from deeptector.data.ffpp import MANIPULATIONS, resolve_ffpp_root
from deeptector.data.sampling import UniformFrameSampler
from deeptector.data.transforms import ImageTransform
from deeptector.utils.device import select_device
from deeptector.utils.seed import seed_everything


def _sample_paths(root: Path, compression: str) -> list[tuple[str, Path]]:
    real_files = sorted(
        (root / "original_sequences" / "youtube" / compression / "videos").glob("*.mp4")
    )
    paths = [("original", real_files[0])]
    for manipulation in MANIPULATIONS:
        directory = root / "manipulated_sequences" / manipulation / compression / "videos"
        paths.append((manipulation, min(directory.glob("*.mp4"))))
    return paths


def main() -> None:
    """Exercise real frames without modifying or re-encoding source videos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/ffpp_m1.yaml")
    parser.add_argument("--ffpp-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="runs/ffpp_real_smoke/report.json")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    data_config = config["data"]
    sampler = UniformFrameSampler(int(data_config.get("frames_per_video", 2)))
    transform = ImageTransform(
        int(data_config.get("input_resolution", 224)),
        str(data_config.get("normalization", "clip")),
    )
    detector = OpenCVHaarFaceDetector()
    reader = OpenCVVideoReader()
    root = resolve_ffpp_root(args.ffpp_root)
    device = select_device(args.device)
    model = build_model(config["model"]).to(device).eval()
    results: list[dict[str, object]] = []
    first_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for category, path in _sample_paths(root, str(data_config.get("compression", "c23"))):
            frame_count = reader.frame_count(str(path))
            fps = reader.frame_rate(str(path))
            indices = sampler.sample(frame_count)
            decoded = failed_decode = faces = failed_faces = 0
            embeddings: list[torch.Tensor] = []
            logits: list[float] = []
            scores: list[float] = []
            for index in indices:
                try:
                    frame = reader.read(str(path), index)
                except OSError:
                    failed_decode += 1
                    continue
                decoded += 1
                box = detector.detect(frame)
                if box is None:
                    failed_faces += 1
                    crop = center_square_crop(frame)
                else:
                    faces += 1
                    crop = detector.crop(frame, box, float(data_config.get("face_margin", 0.2)))
                output = model(transform(crop).unsqueeze(0).to(device))
                embeddings.append(output.embedding.cpu())
                logits.append(float(output.logits.cpu().item()))
                scores.append(float(output.score.cpu().item()))
            if embeddings:
                first_embeddings.append(embeddings[0])
            results.append(
                {
                    "category": category,
                    "video_path": str(path),
                    "frame_count": frame_count,
                    "sampled_indices": indices,
                    "fps": fps,
                    "timestamps_seconds": [index / fps for index in indices],
                    "decoded_frames": decoded,
                    "failed_decodes": failed_decode,
                    "face_detections": faces,
                    "failed_face_detections": failed_faces,
                    "embedding_shape": list(embeddings[0].shape) if embeddings else None,
                    "logits_shape": [len(logits)],
                    "logits": logits,
                    "uncalibrated_scores": scores,
                }
            )

    # Repeat the first selected frame through the exact same deterministic path.
    category, path = _sample_paths(root, str(data_config.get("compression", "c23")))[0]
    index = sampler.sample(reader.frame_count(str(path)))[0]
    frame = reader.read(str(path), index)
    box = detector.detect(frame)
    crop = (
        center_square_crop(frame)
        if box is None
        else detector.crop(frame, box, float(data_config.get("face_margin", 0.2)))
    )
    with torch.no_grad():
        repeated = model(transform(crop).unsqueeze(0).to(device)).embedding.cpu()
    if not first_embeddings:
        raise RuntimeError("No frames were decoded; determinism cannot be verified")
    report = {
        "seed": int(config.get("seed", 42)),
        "preprocessing": data_config,
        "model": {
            "name": config["model"]["name"],
            "freeze_backbone": bool(config["model"].get("freeze_backbone", True)),
            "encoder_parameters_trainable": sum(
                parameter.numel()
                for parameter in model.encoder.parameters()
                if parameter.requires_grad
            ),
        },
        "videos": results,
        "determinism": {
            "category": category,
            "frame_index": index,
            "same_frame_indices": index == results[0]["sampled_indices"][0],
            "embedding_allclose": bool(torch.allclose(first_embeddings[0], repeated, atol=1e-6)),
            "max_absolute_embedding_difference": float(
                torch.max(torch.abs(first_embeddings[0] - repeated)).item()
            ),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
