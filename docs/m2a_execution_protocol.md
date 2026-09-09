# M2A LOMO execution protocol

## Training gate

Each seed-42 LOMO experiment has a separate configuration and run directory. The model,
preprocessing, optimizer, maximum epoch count, early stopping, AMP initial scale, aggregation, and
fixed threshold remain identical to the frozen M1 definition. Only the fold manifest,
`experiment_name`, and explicit LOMO binding metadata differ.

Before `deeptector-train` or `deeptector-evaluate` initializes a model or run directory, the LOMO
binding verifies:

- protocol artifact type and version;
- frozen M1 source-manifest SHA-256 and official-verification basis;
- configured fold slug and held-out manipulation;
- one shared combined manifest for train, validation, and test;
- fold-manifest SHA-256;
- stored split, label, and manipulation counts;
- canonical FF++ video ID, path, compression, and source lineage;
- no duplicate IDs, split leakage, missing files, or invalid fold composition.

Run the CPU/data validation independently with:

```powershell
$env:DEEPTECTOR_DATA_ROOT = 'C:\Users\user\Datasets\deeptector'

deeptector-validate-ffpp-lomo `
  --config configs/experiment/m2a_lomo/deepfakes_seed42.yaml
```

## Bounded XPU preflight

The XPU-only preflight selects one deterministic video for every manipulation category expected in
each split. For one fold this is four train videos, four validation videos, and two held-out test
videos, with the configured eight frames per video. It runs the real decode, face crop, transform,
frozen CLIP encoder, classifier, XPU AMP, loss, and guarded backward path for train/validation. The
held-out test phase is label-free: it checks only decode, forward, finite logits, and device health,
and never records loss, predictions, or metrics. It does not run an epoch and does not write a
checkpoint.

```powershell
deeptector-preflight-lomo `
  --config configs/experiment/m2a_lomo/deepfakes_seed42.yaml `
  --device xpu
```

Reports are written atomically under the ignored fold run directory and record
config/protocol/fold hashes, train/validation losses, frame counts, optimizer-step evidence,
face-detection failures, XPU memory peaks, finite model state, and post-run device survival. Failed
runtime preflights also persist a structured failure report. Missing XPU AMP, optimizer-step, memory,
or survival evidence fails the preflight.

## Full execution order

After all preflights pass, run folds sequentially to avoid competing for the integrated GPU and
shared system memory:

1. Deepfakes
2. Face2Face
3. FaceSwap
4. NeuralTextures

Each uses `runs/<experiment_name>/checkpoint.pt` as the validation-selected best checkpoint and
`last_checkpoint.pt` only for deterministic epoch-boundary resume. Evaluation must use the best
checkpoint from the exact same fold configuration; other-fold and resume checkpoints are rejected.
A fresh invocation is rejected when checkpoint state already exists, and resume must explicitly
name the fold's own `last_checkpoint.pt`. Validation is evaluated first; the held-out test is
evaluated once after model selection and is never used for tuning. Protocol-bound evaluation cannot
override the configured fold manifest.

## Reporting contract

The primary results are video-level ROC-AUC, PR-AUC, balanced accuracy, specificity, and confusion
matrix for each held-out manipulation. Frame-level metrics are secondary. The across-fold mean and
standard deviation describe variation between manipulation tasks; they are not random-seed
confidence intervals. Sigmoid outputs remain uncalibrated scores. These FF++ LOMO results are not
cross-dataset, identity-disjoint, or evidence of universal deepfake detection.
