# DeepTector Milestone 1 Stabilization and Capacity Handoff

> Historical handoff: this snapshot predates the approved batch-8 configuration and deterministic
> epoch-boundary resume implementation. Retain it as provenance, not as the current launch-readiness
> source.

Date: 2026-08-28
Repository: `C:\Users\user\Projects\deeptector`
Branch: `main`
Base HEAD: `ec4ea25 Complete FF++ M1 readiness with Intel XPU support`

## Purpose

This document records the verified state after the final Milestone 1 repository-stabilization and
Intel Arc capacity-validation work. It is a factual handoff for planning the next project task. It
does not authorize a full training run, a commit/push, or implementation of a later milestone.

The primary research objective remains generalization to unseen deepfake manipulations. Milestone 1
is still limited to a reproducible frozen CLIP ViT-B/16 visual baseline.

## Repository state

The task started from a clean `main` working tree at `ec4ea25`. The stabilization work is currently
present as expected, uncommitted local changes. Nothing is staged, committed, or pushed.

The changes cover:

- fail-fast finite checks for training logits, loss, and trainable gradients;
- AMP gradient unscaling before finite-gradient inspection;
- a conservative configurable XPU AMP initial scale of `1024`;
- FF++ directed-pair completeness validation;
- regression tests for numerical failures and missing reverse directed videos;
- hardened generated-artifact ignore rules;
- a separate bounded XPU capacity benchmark CLI, tests, and documentation.

The full M1 experiment configuration still declares `train.batch_size: 32`. The measured and
recommended safe frame batch is `8`; changing the full-run configuration requires an explicit next
task decision.

## Data and manifest verification

The real FF++ c23 manifest was regenerated only as a validation action and remains an ignored,
derived artifact.

- total records: `5000`
- train: `720 real / 2880 fake` (`3600` total)
- validation: `140 real / 560 fake` (`700` total)
- test: `140 real / 560 fake` (`700` total)
- duplicate video IDs: `[]`
- source overlap across splits: `{}`
- missing files: `[]`
- unknown files: `[]`

Completeness now treats `A_B.mp4` and `B_A.mp4` as distinct required manipulated videos for each of
Deepfakes, Face2Face, FaceSwap, and NeuralTextures. Composite source lineage remains represented by
both source IDs for leakage detection.

## Verification results

CPU/default environment:

- Ruff format: passed (`47` files)
- Ruff lint: passed
- pytest: `39 passed, 1 skipped`
- read-only AST parse validation: `44` Python files passed
- `git diff --check`: passed

Intel XPU environment:

- Python environment: `C:\Users\user\Desktop\deeptector_\.venv`
- torch: `2.13.0+xpu`
- `torch.xpu.is_available()`: `True`
- auto-selected device: `xpu`
- device: `Intel(R) Arc(TM) Graphics`
- full pytest: `40 passed`

Git/artifact audit:

- staged files: `0`
- tracked protected artifacts or protected-root files: `0`
- non-ignored untracked files were source/test files only; largest was `30,075` bytes before this
  handoff document was added
- datasets, videos, CLIP weights, checkpoints, runs, caches, and virtual environments remain ignored
- the capacity JSON artifacts exist only below ignored `runs/` directories

## Capacity benchmark conditions

- frozen CLIP ViT-B/16 visual encoder; trainable encoder parameters: `0`
- official FF++ smoke train/validation subset
- seed: `42`
- eight uniformly selected frames per video, flattened into independent frame samples
- resolution: `224 x 224`
- face margin: `0.2`
- CLIP normalization
- train-only balanced sampling
- BCEWithLogitsLoss and the existing XPU AMP/GradScaler path
- frame batches tested: `2`, `4`, and `8` only
- exactly `8` warm-up frames for training and validation per batch
- the next `32` frames measured per repetition
- three independent training and three independent validation repetitions per batch
- identical sampled order, frozen encoder, and cloned initial classifier head state
- XPU synchronization around timing boundaries

The authoritative CLIP input shapes were:

- batch 2: `[2, 3, 224, 224]`
- batch 4: `[4, 3, 224, 224]`
- batch 8: `[8, 3, 224, 224]`

## Capacity results

| Batch | Stable | Train E2E fps | Compute fps | Validation fps | Peak allocated | Peak reserved | Minimum system memory available | Errors |
|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 2 | 3/3 | 4.4867 | 28.1441 | 5.3260 | 344.74 MiB | 386 MiB | 14.68 GiB | none |
| 4 | 3/3 | 4.8185 | 37.9023 | 5.6152 | 352.73 MiB | 402 MiB | 14.69 GiB | none |
| 8 | 3/3 | 4.9634 | 50.3447 | 5.5837 | 378.12 MiB | 422 MiB | 14.66 GiB | none |

All training and validation repetitions completed, all device survival checks passed, and no
non-finite result, OOM, allocation failure, driver failure, or device loss was observed. Batch 8
improved end-to-end training throughput over batch 4 by about `3.01%`, while peak reserved XPU
memory increased by about `4.98%`. Because memory pressure did not increase substantially and all
trials were stable, the benchmark policy selected frame batch `8`. Batch `4` remains a conservative
fallback for long-duration thermal or system-contention concerns.

The authoritative detailed artifact is:

`C:\Users\user\Projects\deeptector\runs\m1_xpu_capacity_20260828T063347Z\capacity_report.json`

SHA-256:

`6704738A75D8F3BB2D93D38013278B45881A03D03F98A4A99A68E3F27ABCC3C8`

An earlier report at `m1_xpu_capacity_20260828T062717Z` was superseded because its Windows process
RSS field was unavailable. Use only the `20260828T063347Z` artifact above.

## Calculated full M1 workload at frame batch 8

- training frame samples per epoch: `28,800`
- validation frame samples per pass: `5,600`
- training optimizer steps per epoch: `3,600`
- validation steps per pass: `700`
- estimated training time per epoch: `1:36:42`
- estimated validation time per epoch: `0:16:43`
- estimated combined epoch time: `1:53:25`
- estimated maximum 20-epoch time: `37:48:28`
- maximum training steps: `72,000`
- maximum validation steps: `14,000`
- early-stopping patience: `5`

These estimates come from bounded 32-frame repetitions. They exclude checkpoint I/O and may be
affected by long-run thermal throttling, power policy, OS contention, and driver behavior. Early
stopping may reduce the actual runtime.

## Current decision boundary

No code, data-integrity, or bounded-XPU blocker remains. Before the full M1 experiment, the next
approved work should decide how to:

1. review and commit the current stabilization diff without including generated artifacts;
2. change the full-run frame batch from `32` to the measured recommendation `8`;
3. perform a final long-run operational preflight;
4. launch, monitor, resume if necessary, and preserve the full M1 training run;
5. evaluate and record the frozen baseline before considering any later milestone.

No full 20-epoch training, full test evaluation, Celeb-DF evaluation, or Milestone 2 implementation
has been performed as part of this handoff.
