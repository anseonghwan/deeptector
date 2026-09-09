# DeepTector

DeepTector is a graduation research project about **generalizable deepfake detection**. Its goal is to learn manipulation-agnostic visual representations that transfer to unseen generators and datasets, rather than memorize artifacts from one benchmark.

## Milestone 1: reproducible visual baseline

This repository currently implements only the first milestone:

The completed FF++ c23 baseline results are recorded in the
[`Milestone 1 final results`](docs/m1_final_results.md) report. Exact environment, configuration,
checkpoint, dataset, and evaluation hashes are preserved in the machine-readable
[`Milestone 1 reproducibility record`](docs/m1_reproducibility.json).

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

## FaceForensics++ real-data integration

Keep FF++ outside this repository and point DeepTector at its parent directory:

```powershell
$env:DEEPTECTOR_DATA_ROOT = "C:\Users\user\Datasets\deeptector"
```

Manifest paths retain `${DEEPTECTOR_DATA_ROOT}/ffpp` rather than embedding a machine-specific
absolute path. The loader expands this variable at runtime. Source videos are never copied,
renamed, re-encoded, or modified.

DeepTector does not invent an FF++ split. Obtain the official FaceForensics++ split metadata and
place these three files together in a directory selected by the researcher:

```text
train.json
val.json
test.json
```

Each file must contain the official JSON list of two-source pairs. Generate the combined manifest
and mandatory pre-training split report with:

```powershell
deeptector-prepare-ffpp `
  --splits-dir metadata/ffpp/splits `
  --output data/manifests/ffpp_c23.csv `
  --smoke-output data/manifests/ffpp_c23_smoke.csv `
  --report data/manifests/ffpp_c23_split_report.json
```

Generation fails if official files are missing or malformed; either member of a manipulated pair
crosses splits; expected real/manipulated files are missing; unknown videos exist; or video IDs are
duplicated. For a fake such as `000_003.mp4`, the manifest records `source_video_id=000|003`, and
both sources participate independently in leakage validation.

Download `openai/clip-vit-base-patch16` once to
`${DEEPTECTOR_DATA_ROOT}/models/openai-clip-vit-base-patch16`, then run the bounded real-data path
test:

```powershell
deeptector-real-smoke --config configs/experiment/ffpp_m1.yaml --device auto
```

The report records actual FPS-derived timestamps, decode failures, face-detection failures,
embedding/logit shapes, uncalibrated scores, and repeated-pass embedding consistency.

The official FF++ config enables deterministic, train-only class balancing. A PyTorch
`WeightedRandomSampler` derives inverse-frequency weights from the current train manifest and uses
the experiment seed. It keeps the number of samples per epoch unchanged. Validation and test never
use this sampler and retain the official 1:4 real:fake distribution. Set
`train.balanced_sampling: false` to recover the original shuffled train loader.

Only after manifest and smoke validation pass, launch the full frozen-backbone baseline:

```powershell
deeptector-train --config configs/experiment/ffpp_smoke.yaml --device auto
deeptector-evaluate --config configs/experiment/ffpp_smoke.yaml `
  --split validation `
  --checkpoint runs/ffpp_clip_frozen_smoke/checkpoint.pt

deeptector-train --config configs/experiment/ffpp_m1.yaml --device auto
deeptector-evaluate --config configs/experiment/ffpp_m1.yaml `
  --split validation `
  --checkpoint runs/ffpp_clip_frozen_m1/checkpoint.pt
deeptector-evaluate --config configs/experiment/ffpp_m1.yaml `
  --split test `
  --checkpoint runs/ffpp_clip_frozen_m1/checkpoint.pt
```

Training writes two distinct checkpoint contracts. `checkpoint.pt` is replaced only when
validation loss improves and is the checkpoint used for evaluation. `last_checkpoint.pt` is
atomically replaced after every successfully completed epoch and contains the optimizer,
GradScaler, early-stopping, RNG, and balanced-sampler state required for deterministic
epoch-boundary recovery. Resume into the same run directory with:

```powershell
deeptector-train --config configs/experiment/ffpp_m1.yaml --device auto `
  --resume-from runs/ffpp_clip_frozen_m1/last_checkpoint.pt
```

The configured epoch count remains the total maximum; it is not interpreted as additional epochs.
Resume does not support recovery from the middle of an epoch.

Before launching the full experiment on an Intel Arc integrated GPU, run the bounded capacity
benchmark. It keeps the full eight-frame M1 preprocessing definition but measures only the official
smoke subset with frame batch sizes 2, 4, and 8. Reports are written under ignored `runs/` paths.

```powershell
deeptector-benchmark-xpu --config configs/experiment/ffpp_m1.yaml --device auto
```

The benchmark records end-to-end and compute-only throughput, validation throughput, finite
logit/loss/gradient checks, XPU allocator peaks, shared system-memory evidence, face-detection
failures, and post-trial device survival. It never launches the full 20-epoch experiment.

For Celeb-DF v2, create a test-only manifest from its official testing list and pass it with
`--manifest`; never include Celeb-DF records in the M1 training or validation manifests.

## M2A protocol infrastructure: FF++ c23 LOMO

The deterministic Leave-One-Manipulation-Out generator creates four protocol folds from the
existing FF++ c23 manifest. Each fold excludes one of `Deepfakes`, `Face2Face`, `FaceSwap`, or
`NeuralTextures` from train and validation, then tests on official-test real videos plus only that
held-out fake method. Original split assignments and portable dataset paths are preserved.

```powershell
deeptector-prepare-ffpp-lomo `
  --source-manifest data/manifests/ffpp_c23.csv `
  --output-dir data/manifests/ffpp_lomo_c23
```

Outputs stay under ignored `data/` paths. The generated `protocol.json` records source and fold
hashes, counts, and leakage checks. This protocol tests held-out manipulation transfer within
FF++; it is not cross-dataset or identity-disjoint evaluation, and it cannot establish universal
manipulation-agnostic detection. See
[`docs/m2a_lomo_protocol.md`](docs/m2a_lomo_protocol.md) for the exact contract and future training
handoff. No LOMO training is started by generation.

Seed-42 execution configs live under `configs/experiment/m2a_lomo/`. Every LOMO train/evaluate
entry validates the configured protocol version, frozen source hash, fold hash, held-out method,
and counts before model initialization. Validate or run the bounded real-XPU preflight with:

```powershell
deeptector-validate-ffpp-lomo `
  --config configs/experiment/m2a_lomo/deepfakes_seed42.yaml

deeptector-preflight-lomo `
  --config configs/experiment/m2a_lomo/deepfakes_seed42.yaml `
  --device xpu
```

The XPU-only preflight processes one video per required split/manipulation category and writes no
checkpoint. Its held-out test pass is label-free and records no loss or prediction metric. See
[`docs/m2a_execution_protocol.md`](docs/m2a_execution_protocol.md) for the execution, resume,
evaluation, and reporting contract.

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
├── config.yaml
├── checkpoint.pt
├── last_checkpoint.pt
├── training.log
└── evaluations/<split>/<run-id>/
    ├── artifact_manifest.json
    ├── config.yaml
    ├── metrics.json
    ├── frame_predictions.csv
    └── video_predictions.csv
```

Evaluation IDs default to UTC timestamps. Pass `--evaluation-id` for a stable explicit identifier.
An existing non-empty evaluation directory is rejected instead of silently overwritten. The metrics
artifact records the evaluated split, resolved checkpoint, aggregation, threshold, UTC timestamp,
label counts, face-detection failures, and config reference.

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
