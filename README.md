# DeepTector

DeepTector is a graduation research project about **generalizable deepfake detection**. Its goal is to learn manipulation-agnostic visual representations that transfer to unseen generators and datasets, rather than memorize artifacts from one benchmark.

## Milestone 1: reproducible visual baseline

This repository currently implements only the first milestone:

```text
video → deterministic uniform frame sampling → face detection/crop
      → frozen CLIP ViT-B/16 visual encoder → LayerNorm + binary head
      → frame scores → configurable video aggregation → metrics
```

The default encoder is `openai/clip-vit-base-patch16`. Its visual tower is frozen and the classifier head is trained with `BCEWithLogitsLoss`. Model outputs include the embedding, logit, and sigmoid score. The score is **not a calibrated probability**.

No dataset or pretrained weight is downloaded by setup or tests. CLI configuration defaults to `local_files_only: true`; cache the CLIP model separately in an environment where its license and storage requirements have been reviewed, or explicitly change that setting for an intentional download.

## Installation

Python 3.10+ is supported. The project uses `pyproject.toml` and standard Python packaging:

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
python -m pip install -e ".[dev]"
```

## Manifest data format

Training code consumes CSV, JSON, or JSONL manifests rather than dataset-specific directory layouts. Required fields are:

| Field | Meaning |
|---|---|
| `video_id` | Unique video identifier |
| `video_path` | Relative or absolute path supplied by the researcher |
| `dataset` | Dataset name, such as `ffpp` or `celeb_df` |
| `split` | `train`, `validation`/`val`, or `test` |
| `label` | `0` REAL, `1` FAKE |

Optional fields are `identity_id`, `source_video_id`, `manipulation_type`, `manipulation_family`, `compression`, `fps`, and `duration`. Missing optional values are safe. `source_video_id` should identify the pristine source lineage shared by related real/manipulated videos when known.

Example:

```csv
video_id,video_path,dataset,split,label,identity_id,source_video_id,manipulation_type,compression,fps,duration
001-real,/data/ffpp/original/001.mp4,ffpp,train,0,p001,001,real,c23,30,8.2
001-df,/data/ffpp/deepfakes/001.mp4,ffpp,train,1,p001,001,Deepfakes,c23,30,8.2
```

Splits must be assigned before frame extraction. Manifest loading rejects a `video_id` or non-empty `source_video_id` that crosses splits. For identity-disjoint research, call `assert_no_split_leakage(records, identity_disjoint=True)` after assigning every available `identity_id`.

### Converting public datasets

Conversion is metadata indexing, not downloading:

- **FaceForensics++:** enumerate original and manipulation video folders; record the manipulation folder name and compression (`c0`, `c23`, `c40`); use the published source pair/index as `source_video_id` and keep its entire lineage in one split.
- **Celeb-DF:** translate the official real/fake list to labels; derive stable IDs from relative paths; populate identities only when a verified identity mapping is available.
- **DFDC:** join video filenames with each part's `metadata.json`; map `REAL`/`FAKE` to `0`/`1`; preserve `original` as `source_video_id`; split all videos with the same original together.
- **DF40:** retain the method and family metadata exposed by the dataset index; group related source videos and identities before assigning splits.

Never infer identity from a filename unless that convention is guaranteed by the dataset documentation. Keep dataset licenses and paths outside this repository.

## Configuration and commands

Copy and edit [`configs/experiment/baseline.yaml`](configs/experiment/baseline.yaml). All sampling, preprocessing, optimization, aggregation, and seed choices live in the effective config.

Train:

```bash
deeptector-train --config configs/experiment/baseline.yaml --device auto
```

Evaluate the configured test manifest:

```bash
deeptector-evaluate --config configs/experiment/baseline.yaml \
  --checkpoint runs/clip_frozen_baseline/checkpoint.pt
```

Cross-dataset evaluation (train on dataset A, test on a test-only manifest for B):

```bash
deeptector-evaluate --config configs/experiment/baseline.yaml \
  --checkpoint runs/clip_frozen_baseline/checkpoint.pt \
  --manifest data/dataset_b_test.csv \
  --run-dir runs/cross_dataset
```

Run offline tests and lint:

```bash
pytest
ruff check .
```

## Evaluation and reproducibility

The evaluator reports frame- and video-level ROC-AUC, PR-AUC, accuracy, precision, recall, F1, and a `[[TN, FP], [FN, TP]]` confusion matrix. ROC-AUC and PR-AUC are `null` when a split contains only one class. Video score aggregation is configurable as `mean`, `max`, or `top_k_mean`.

Each run directory contains:

```text
runs/<experiment>/
├── config.yaml          # effective configuration
├── checkpoint.pt       # best validation checkpoint
├── training.log        # epoch-level history
├── metrics.json        # frame/video metrics and experiment provenance
└── predictions.csv     # dataset, video, frame, label, logit, score, face status
```

Seeds cover Python, NumPy, CPU PyTorch, and CUDA PyTorch. Deterministic PyTorch algorithms are requested where practical. Uniform frame indices and face crops are deterministic. A missing face uses a deterministic center crop and is explicitly marked in predictions; the evaluator reports the failure rate.

## Source layout

- `data`: manifest validation, leakage guards, sampling, video reading, face crops, transforms
- `models`: replaceable `VisualEncoder`, CLIP adapter, embedding-preserving classifier
- `training`: loss, optimizer, trainer, checkpoint lifecycle
- `evaluation`: binary metrics, video aggregation, predictions/results persistence
- `cli`: configuration-driven train and evaluation commands

## Future milestones (not implemented)

- M2: SBI and realistic augmentation
- M3: spatial-frequency fusion
- M4: contrastive/manipulation-agnostic representation learning
- M5: temporal clip modeling
- M6: adaptive cascade inference
- M7: probability calibration
- M8: audio-visual branch
- M9: Vector DB forensic retrieval and hard-negative mining
- M10: local VLM/LLM evidence explanation

Experimental deepfake-detection performance is intentionally not reported here; it must come from controlled runs on properly licensed real datasets.
