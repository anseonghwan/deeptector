# Milestone 1 Real-Data Validation Report

Date: 2026-08-21

Historical assessment (2026-08-21): **PASS WITH CONDITIONS**

Current full-baseline launch readiness (2026-08-25): **PASS FOR FULL M1 BASELINE**

The real FF++ decode, face-crop, CLIP embedding, classifier, and deterministic replay path works.
Official FF++ split metadata is not present, so source-safe manifest generation and real-data
training/evaluation were deliberately blocked. This is not an experimental performance report.

## 1. Repository audit summary

The original M1 repository already contained a dataset-independent manifest loader, deterministic
uniform frame sampler, OpenCV video reader and Haar face detector, modular CLIP visual encoder,
embedding-preserving binary head, BCEWithLogits trainer, checkpointing, frame/video metrics,
configurable aggregation, and train/evaluate CLIs. It did not contain a manifest generator, FF++
parser, official split importer/report, composite-source leakage handling, or real-data smoke CLI.

## 2. Files changed and why

- `src/deeptector/data/ffpp.py`: official split import, FF++ discovery, manifest/report generation.
- `src/deeptector/data/manifest.py`: environment path expansion, duplicate IDs, composite sources.
- `src/deeptector/data/dataset.py`: actual container FPS access.
- `src/deeptector/cli/prepare_ffpp.py`: fail-closed manifest and split-report command.
- `src/deeptector/cli/real_smoke.py`: bounded decode/face/CLIP/determinism validation.
- `configs/experiment/ffpp_m1.yaml`: full M1 configuration.
- `configs/experiment/ffpp_smoke.yaml`: bounded one-epoch configuration.
- `pyproject.toml`: new CLI commands and OpenCV 4.x compatibility bound.
- `tests/test_ffpp.py` and related tests: official split, path, composite leakage coverage.
- `README.md`: setup, required official metadata, commands, and Celeb-DF constraint.

## 3. Dataset discovery result

`DEEPTECTOR_DATA_ROOT=C:\Users\user\Datasets\deeptector` was present. FF++ c23 discovery found:

| Category | MP4 count | Malformed filename count |
|---|---:|---:|
| Original | 1,000 | 0 |
| Deepfakes | 1,000 | 0 |
| Face2Face | 1,000 | 0 |
| FaceSwap | 1,000 | 0 |
| NeuralTextures | 1,000 | 0 |
| Total | 5,000 | 0 |

No JSON, CSV, or TXT official split metadata was found beneath the FF++ root.

## 4. Manifest statistics

Not produced. `deeptector-prepare-ffpp` correctly stopped before writing a manifest because
`train.json`, `val.json`, and `test.json` were absent. It did not substitute a random split.

## 5. Split statistics

Unavailable until the three official pair-list JSON files are supplied. The implemented report will
record real/fake counts per split and manipulation counts per split before training is permitted.

## 6. Leakage checks

The importer maps each official pair and each constituent source to a split. A manipulated filename
such as `000_003.mp4` is represented as `source_video_id=000|003`; both `000` and `003` are checked
independently. Duplicate video IDs, source overlap, missing expected files, unknown files, malformed
metadata, or missing official split files block manifest generation/training.

## 7. Real-data decoding results

One video from each of the five categories was sampled with 8 uniform frames:

| Category | Frame count | FPS | Sampled indices | Decoded | Failed |
|---|---:|---:|---|---:|---:|
| Original | 396 | 25 | 0, 56, 112, 169, 225, 282, 338, 395 | 8 | 0 |
| Deepfakes | 396 | 25 | 0, 56, 112, 169, 225, 282, 338, 395 | 8 | 0 |
| Face2Face | 303 | 25 | 0, 43, 86, 129, 172, 215, 258, 302 | 8 | 0 |
| FaceSwap | 303 | 25 | 0, 43, 86, 129, 172, 215, 258, 302 | 8 | 0 |
| NeuralTextures | 303 | 25 | 0, 43, 86, 129, 172, 215, 258, 302 | 8 | 0 |

The report artifact is `runs/ffpp_real_smoke/report.json`. Every embedding had shape `[1, 768]`;
each video produced 8 scalar logits and 8 uncalibrated sigmoid scores. These scores come from an
untrained random classification head and are not performance measurements.

## 8. Face-detection failure statistics

40 of 40 decoded frames produced Haar face detections. Failure count: 0. Failure rate: 0.0%.

## 9. Determinism test results

The original video selected frame 0 on both passes. With the frozen encoder in eval mode and the
same preprocessing, embeddings were exactly equal within `atol=1e-6`; maximum absolute difference
was `0.0`.

## 10. Small training smoke-test result

**Blocked intentionally.** Running a real-data training split before importing the official FF++
split would violate the experiment rules. Synthetic trainer tests pass, but they are not treated as
real-data acceptance evidence.

## 11. Exact command/config for full FF++ baseline

After placing official files in `metadata/ffpp/splits`:

```powershell
$env:DEEPTECTOR_DATA_ROOT = "C:\Users\user\Datasets\deeptector"
deeptector-prepare-ffpp --splits-dir metadata/ffpp/splits
deeptector-train --config configs/experiment/ffpp_m1.yaml --device auto
deeptector-evaluate --config configs/experiment/ffpp_m1.yaml `
  --split validation `
  --checkpoint runs/ffpp_clip_frozen_m1/checkpoint.pt
deeptector-evaluate --config configs/experiment/ffpp_m1.yaml `
  --split test `
  --checkpoint runs/ffpp_clip_frozen_m1/checkpoint.pt
```

The full config records seed 42, frozen CLIP ViT-B/16, 8 frames/video, 224 resolution, 0.2 face
margin, CLIP normalization, mean video aggregation, batch size 32, AdamW, learning rate 0.001,
20 maximum epochs, patience 5, checkpoint and prediction locations under `runs/`.

## 12. Remaining blockers

1. Supply official FF++ `train.json`, `val.json`, and `test.json` pair lists.
2. Generate and inspect the manifest/split report.
3. Run bounded real-data training, validation, prediction export, and metrics.
4. Only then launch the full baseline and evaluate validation/test separately.
5. Celeb-DF v2 remains unavailable; no Celeb-DF metrics were generated.

## 13. Acceptance decision

**PASS WITH CONDITIONS.** Real video preprocessing and the frozen visual foundation encoder are
validated, but M1 cannot be an unconditional PASS until official split metadata enables a leakage-
safe real training/evaluation run.

## 14. Readiness update — 2026-08-25

This section preserves the historical findings above and records what changed afterward. It is a
launch-readiness decision, not a declaration that Milestone 1 experimentation is complete.

### Official split blocker — RESOLVED

The official `train.json`, `val.json`, and `test.json` files were added under
`metadata/ffpp/splits`. `deeptector-prepare-ffpp` completed successfully and generated 5,000
records:

| Split | Real | Fake | Total | Each fake manipulation |
|---|---:|---:|---:|---:|
| Train | 720 | 2,880 | 3,600 | 720 |
| Validation | 140 | 560 | 700 | 140 |
| Test | 140 | 560 | 700 | 140 |

Duplicate video IDs: 0. Source overlap: 0. Missing files: 0. Unknown files: 0. Each split also
contains the stated number of original videos.

### Evaluation artifact contract

Each evaluation now writes to:

```text
runs/<experiment>/evaluations/<split>/<run-id>/
```

The directory contains `artifact_manifest.json`, `config.yaml`, `metrics.json`,
`frame_predictions.csv`, and `video_predictions.csv`. The metrics include frame/video ROC-AUC,
PR-AUC, accuracy, precision, recall, F1, confusion matrix, aggregation, threshold, label counts,
checkpoint identity, UTC timestamp, and face-detection counts. Existing non-empty evaluation IDs are
rejected, preventing silent overwrite.

### Train-only balancing policy

The full FF++ and smoke configs enable `train.balanced_sampling: true`. Train sample weights are
computed from the actual manifest counts as inverse class frequency and passed to a seeded
`WeightedRandomSampler`. The effective samples per epoch remain equal to the unbalanced dataset
length. `UniformFrameSampler` remains an independent temporal sampler. Validation and test are never
rebalanced; requesting balancing outside train raises an error. Setting the option to false restores
the previous train shuffle behavior.

### New bounded real-data smoke

Run root:

```text
runs/m1_readiness_20260825/ffpp_clip_frozen_smoke/
```

- epoch: 1
- train loss: 0.694899
- validation loss: 1.114771
- train face detection: 5 attempted, 0 failed
- validation face detection: 5 attempted, 0 failed
- checkpoint saved and reload verified
- checkpoint config: frozen backbone `true`, balanced sampling `true`
- train loader: `{real: 1, fake: 4}`, balancing enabled
- validation loader: `{real: 1, fake: 4}`, balancing disabled

Validation evaluation artifacts:

```text
runs/m1_readiness_20260825/ffpp_clip_frozen_smoke/
  evaluations/validation/validation_smoke_balanced_v1/
```

All five frame and five video predictions were finite and persisted. Face failures were 0/5.
Frame/video metrics were identical because the smoke config sampled one frame per video: ROC-AUC
0.75, PR-AUC 0.95, accuracy 0.20, precision/recall/F1 0.0, confusion matrix
`[[1, 0], [4, 0]]`. These five-video smoke metrics are pipeline evidence only and are not M1
performance claims.

### Verification

- pytest: 25 passed
- Ruff lint: passed
- Ruff format check: passed
- Python compile: passed
- git diff check: passed
- NaN/Inf guard: implemented and no non-finite smoke outputs observed
- CLIP frozen: verified (`freeze_backbone=true`, zero trainable encoder parameters in real smoke)
- official validation/test distributions: unchanged at 140 real / 560 fake each

### Remaining work

No launch-readiness blocker remains. The 20-epoch full FF++ experiment still requires explicit user
approval and has not been started. Celeb-DF remains outside M1 training and no Celeb-DF metrics were
fabricated.

### Updated decision

**PASS FOR FULL M1 BASELINE.** All requested launch gates have passed. This permits launching the
full experiment; it does not declare Milestone 1 complete or establish detector performance.
