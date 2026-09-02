# DeepTector Milestone 1 Final Results

Date: 2026-09-02

Decision: **COMPLETE FOR THE IN-DATASET FF++ C23 BASELINE**

This report freezes the final result of the first DeepTector milestone: a reproducible visual
baseline trained and evaluated on leakage-checked FaceForensics++ (FF++) c23 splits. It does not
establish generalization to unseen manipulations, identities, generators, or datasets. The
machine-readable source of truth is [`m1_reproducibility.json`](m1_reproducibility.json).

## 1. Experimental scope

The baseline uses this pipeline:

```text
video → 8 deterministic uniform frames → face detection/crop → CLIP normalization
      → frozen CLIP ViT-B/16 vision tower → LayerNorm + linear binary head
      → frame scores → mean video aggregation → binary metrics
```

- Dataset: FaceForensics++ c23
- Manipulations: Deepfakes, Face2Face, FaceSwap, NeuralTextures
- Encoder: locally cached `openai/clip-vit-base-patch16`, frozen
- Trainable component: LayerNorm plus one linear binary classification layer
- Seed: `42`
- Maximum epochs: `20`
- Early-stopping patience: `5`
- Batch size: `8` frames
- Optimizer: AdamW, learning rate `0.001`, weight decay `0.0001`
- Training sampling: inverse-manifest-label-frequency weighted sampling
- Validation/test sampling: original unbalanced distributions
- AMP: XPU float16 with GradScaler, initial scale `1024`
- Evaluation aggregation: mean across 8 frame scores per video
- Evaluation threshold: fixed at `0.5`; it was not changed after observing test results

Sigmoid scores are uncalibrated scores, not calibrated probabilities.

Loading the full pretrained CLIP checkpoint through `CLIPVisionModel` reports unused text-tower and
projection keys as `UNEXPECTED`. This is expected for the intentionally vision-only architecture;
the vision weights loaded and the warning was not a training or evaluation failure.

## 2. Data integrity

The generated manifest contains 5,000 records. The official FF++ source-pair split metadata is
checked for both directed manipulated videos (`A_B` and `B_A`).

| Split | Real | Fake | Total | Each fake manipulation |
|---|---:|---:|---:|---:|
| Train | 720 | 2,880 | 3,600 | 720 |
| Validation | 140 | 560 | 700 | 140 |
| Test | 140 | 560 | 700 | 140 |

Integrity report:

- duplicate video IDs: `0`
- source overlap across splits: `0`
- missing files: `0`
- unknown files: `0`
- identity-disjoint enforcement: `false` because verified identity metadata was unavailable

The split is source-lineage safe, but this result must not be presented as identity-disjoint.

## 3. Execution environment

- Host: Lenovo model `83D4`
- OS: Windows 11 Home, build `26200`, 64-bit
- CPU: Intel Core Ultra 7 155H
- Memory: `33,945,935,872` bytes
- GPU: integrated Intel Arc Graphics
- Windows graphics driver: `32.0.101.8974`
- oneAPI Level-Zero runtime driver: `1.15.39183+1`
- Python: `3.11.9`
- PyTorch: `2.13.0+xpu`
- Transformers: `5.15.1`
- OpenCV: `4.14.0`

The XPU capacity benchmark selected batch 8 as the largest measured stable batch. Its bounded
batch-8 result was 4.9634 end-to-end training frames/s, 50.3447 compute frames/s, 5.5837 validation
frames/s, 378.12 MiB peak allocated XPU memory, and 422 MiB peak reserved XPU memory. A separate
448-step operational preflight ran for 716.22 seconds without OOM, device loss, non-finite values,
or face-detection failures.

## 4. Training and recovery record

Epochs 1-3 were produced at commit `1a3370f5d11682c8529e49c440dd4ed124283716`. The first epoch-4
attempt stopped when the finite-gradient guard detected a non-finite gradient in `head.1.weight`.
Because only successful epoch boundaries are persisted, the partial epoch-4 optimizer updates were
discarded and `last_checkpoint.pt` remained at the finite epoch-3 boundary.

Commit `b1af2d1a968fcb8d7352d39e71f041f0b27dc505` added tested AMP overflow recovery: GradScaler skips
the affected optimizer update, reduces the scale, logs the event, and continues. Training then
resumed deterministically from epoch 3 and reran epoch 4 from its beginning.

The persistent 12-epoch trajectory contains 43,200 nominal training batches and 43,185 optimizer
updates. Fifteen recoverable AMP overflow events skipped fifteen updates. Thirteen events backed off
from `65536` to `32768`; two backed off from `32768` to `16384`. All saved model and optimizer
tensors remained finite.

| Epoch | Train loss | Validation loss | Best so far |
|---:|---:|---:|:---:|
| 1 | 0.476372 | 0.450362 | yes |
| 2 | 0.415328 | 0.397710 | yes |
| 3 | 0.390271 | 0.375049 | yes |
| 4 | 0.379208 | 0.427683 | no |
| 5 | 0.375197 | 0.424975 | no |
| 6 | 0.369092 | 0.422914 | no |
| 7 | 0.362156 | **0.345812** | **yes** |
| 8 | 0.363870 | 0.535943 | no |
| 9 | 0.350221 | 0.423961 | no |
| 10 | 0.352544 | 0.451690 | no |
| 11 | 0.356799 | 0.438044 | no |
| 12 | 0.346724 | 0.481185 | no |

Epoch 12 completed the fifth consecutive non-improving epoch after epoch 7, so early stopping ended
the run normally. The evaluation checkpoint is `checkpoint.pt` from epoch 7. The epoch-12
`last_checkpoint.pt` is a recovery artifact and must not be used for final evaluation.

The face statistics logged at training completion cover only the resumed epoch 4-12 process:

- train: 1,376 failures / 259,200 attempts (`0.5309%`)
- validation: 351 failures / 50,400 attempts (`0.6964%`)

Epochs 1-3 completed in the earlier process, whose accumulated face counters were not persisted.

## 5. Final evaluation

Both splits were evaluated once with the epoch-7 best checkpoint, mean video aggregation, and the
predefined threshold `0.5`.

### Frame-level metrics

| Metric | Validation | Test | Test - validation |
|---|---:|---:|---:|
| ROC-AUC | 0.883114 | 0.874245 | -0.008868 |
| PR-AUC | 0.967341 | 0.965126 | -0.002214 |
| Accuracy | 0.843571 | 0.828214 | -0.015357 |
| Precision | 0.908617 | 0.915643 | +0.007026 |
| Recall | 0.894420 | 0.864955 | -0.029464 |
| F1 | 0.901462 | 0.889578 | -0.011885 |
| Specificity | 0.640179 | 0.681250 | +0.041071 |
| Balanced accuracy | 0.767299 | 0.773103 | +0.005804 |

Validation confusion matrix: `[[717, 403], [473, 4007]]`.

Test confusion matrix: `[[763, 357], [605, 3875]]`.

### Video-level metrics

| Metric | Validation | Test | Test - validation |
|---|---:|---:|---:|
| ROC-AUC | 0.921977 | **0.914885** | -0.007092 |
| PR-AUC | 0.977895 | **0.977209** | -0.000686 |
| Accuracy | 0.878571 | **0.864286** | -0.014286 |
| Precision | 0.914485 | **0.928177** | +0.013692 |
| Recall | 0.935714 | **0.900000** | -0.035714 |
| F1 | 0.924978 | **0.913871** | -0.011107 |
| Specificity | 0.650000 | **0.721429** | +0.071429 |
| Balanced accuracy | 0.792857 | **0.810714** | +0.017857 |

Validation confusion matrix: `[[91, 49], [36, 524]]`.

Test confusion matrix: `[[101, 39], [56, 504]]`.

The video ROC-AUC gap is `-0.007092`, so the FF++ held-out test result is close to validation. The
test distribution still contains four times as many fake videos as real videos; ROC-AUC, balanced
accuracy, and the confusion matrix are therefore more informative than accuracy alone. The model's
test false-positive rate is `27.86%` and remains an important limitation.

Evaluation face-detection failures were 39/5,600 (`0.6964%`) on validation and 53/5,600 (`0.9464%`)
on test. All 11,200 frame scores and all 1,400 video scores were finite and within `[0, 1]`.

## 6. Reproducibility and artifact identity

Primary input identities:

| Artifact | SHA-256 |
|---|---|
| M1 config | `3d197f9d7a179266207b10e9071a16e0efabb34421a33d4531ef3f6f56fa3910` |
| FF++ manifest | `ebee8dbfd4ce9dd23228a2063a8adf22dfe2e4c740a19a12ba6f7ea63eea2ab2` |
| Split report | `6cd1d6c87211cf1233529533d86f966c448586f64d2227193a9ccec59b32851c` |
| Official train split | `e59386911255e6fb0a79a7808ec2210536b8c0474268342694c4bca766d1b347` |
| Official validation split | `b48cc511f66938e05356aaa9c67e150b1e3638c355db3d745225b0022884b36e` |
| Official test split | `886f5a0da623c25820692e0d8dc33d197ddb1db527a7f1cfcb9bcbca60fe4f40` |
| CLIP PyTorch weights | `ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0` |

Primary local result artifacts:

| Artifact | SHA-256 |
|---|---|
| Training log | `99b409929930de26445b6dfb759fdfb0e437985ad8f5ba1815eb17fbecbed7c1` |
| Epoch-7 best checkpoint | `9cae6b20acac7883fb99400f537d3fc75a77e60cf7990cbf052c41e89ccca517` |
| Epoch-12 last checkpoint | `4cc32cab17ba2dcd3b27a34e099f2d0ce8a4c6b69f7e0744c9a9eabbb96716da` |
| Validation metrics | `c333eede673799e516d53233c091b1201461f2954ff2b0933f7a29c0c537796b` |
| Test metrics | `96d124af0b8223dc20e9f93380389d5e78c537f7a82d74528e4945e9ecf35c02` |

Dataset files, pretrained weights, checkpoints, prediction CSVs, and `runs/` remain intentionally
ignored by Git. The JSON record includes hashes for every evaluation file so local artifacts can be
verified without committing large or licensed files.

## 7. Exact execution commands

All commands were run from the repository root with the same XPU-enabled Python environment.

Initial training:

```powershell
$env:DEEPTECTOR_DATA_ROOT = 'C:\Users\user\Datasets\deeptector'
& 'C:\Users\user\Desktop\deeptector_\.venv\Scripts\python.exe' `
  -m deeptector.cli.train `
  --config 'configs/experiment/ffpp_m1.yaml' `
  --device xpu `
  --run-dir 'runs'
```

Epoch-boundary resume after the AMP recovery change:

```powershell
& 'C:\Users\user\Desktop\deeptector_\.venv\Scripts\python.exe' `
  -m deeptector.cli.train `
  --config 'configs/experiment/ffpp_m1.yaml' `
  --device xpu `
  --run-dir 'runs' `
  --resume-from 'runs/ffpp_clip_frozen_m1/last_checkpoint.pt'
```

Validation and test evaluation used the same best checkpoint:

```powershell
& 'C:\Users\user\Desktop\deeptector_\.venv\Scripts\python.exe' `
  -m deeptector.cli.evaluate `
  --config 'configs/experiment/ffpp_m1.yaml' `
  --device xpu `
  --split validation `
  --checkpoint 'runs/ffpp_clip_frozen_m1/checkpoint.pt' `
  --run-dir 'runs'

& 'C:\Users\user\Desktop\deeptector_\.venv\Scripts\python.exe' `
  -m deeptector.cli.evaluate `
  --config 'configs/experiment/ffpp_m1.yaml' `
  --device xpu `
  --split test `
  --checkpoint 'runs/ffpp_clip_frozen_m1/checkpoint.pt' `
  --run-dir 'runs'
```

## 8. Verification status

After the AMP recovery change:

- forced real-XPU AMP overflow recovery test: `1 passed`
- full CPU/XPU pytest suite: `50 passed`
- Ruff format: `45 files left unchanged`
- Ruff lint: passed
- best and last checkpoint tensors: finite
- validation/test artifact row counts, label counts, score bounds, and unique-video counts: passed

## 9. Interpretation and next research gate

M1 establishes a functioning frozen visual baseline with video ROC-AUC `0.914885` on the internal
FF++ c23 test split. It does not yet answer the project's main question of manipulation-agnostic
generalization. No Celeb-DF, DFDC, DF40, leave-one-manipulation-out, identity-disjoint, calibrated
threshold, or confidence-interval result is claimed.

The next research gate should preserve this M1 result unchanged and evaluate cross-dataset or
leave-one-manipulation-out transfer under a separately approved protocol. Test results from this run
must not be used to retune M1.
