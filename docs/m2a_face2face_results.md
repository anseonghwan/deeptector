# M2A LOMO Face2Face fold result

## Status

- Fold: Face2Face held out
- Seed: 42
- Status: complete
- Training device: Intel XPU (`xpu`)
- Best checkpoint: epoch 6 (`validation_loss=0.367053`)
- Stopping: early stopping at epoch 11 (`patience=5`)
- Held-out test policy: one completed evaluation after validation acceptance

This fold trained on original, Deepfakes, FaceSwap, and NeuralTextures videos while
reserving Face2Face for the final held-out test. Frozen M1 inputs and artifacts were
not changed.

## Evaluation results

| Split | Level | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Specificity | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Frame | 0.892177 | 0.961275 | 0.839955 | 0.907242 | 0.876190 | 0.891446 | 0.731250 | 0.803720 |
| Validation | Video | 0.926837 | 0.973834 | 0.878571 | 0.925121 | 0.911905 | 0.918465 | 0.778571 | 0.845238 |
| Held-out test | Frame | 0.797198 | 0.801260 | 0.712946 | 0.706852 | 0.727679 | 0.717114 | 0.698214 | 0.712946 |
| Held-out test | Video | 0.843724 | 0.846825 | 0.746429 | 0.734694 | 0.771429 | 0.752613 | 0.721429 | 0.746429 |

Confusion matrices use `[[TN, FP], [FN, TP]]`:

- Validation frame: `[[819, 301], [416, 2944]]` (4,480 frames)
- Validation video: `[[109, 31], [37, 383]]` (560 videos)
- Held-out test frame: `[[782, 338], [305, 815]]` (2,240 frames)
- Held-out test video: `[[101, 39], [32, 108]]` (280 videos)

## Execution observations

Training completed without interruption. Nine AMP gradient-overflow events were
handled by reducing the scaler and skipping only the affected optimizer step. Train
face detection attempted 253,440 frames and failed on 1,480 (0.5840%); validation
attempted 49,280 and failed on 363 (0.7366%). Evaluation face-detection failures were
33/4,480 (0.7366%) for validation and 16/2,240 (0.7143%) for held-out test.

The first held-out test launcher exited before evaluating data and produced no result
directory, predictions, metrics, or traceback. After confirming that no XPU process
or test artifact remained, the same immutable best checkpoint was relaunched. That
single completed test evaluation is the result reported here; it was not used for
tuning.

## Artifacts

- Run directory: `runs/m2a_lomo_face2face_seed42`
- Validation evaluation: `evaluations/validation/20260910T214445193900Z`
- Held-out test evaluation: `evaluations/test/20260911T002547089715Z`
- Reproducibility record: `docs/m2a_face2face_reproducibility.json`

The machine-readable record contains exact artifact hashes, metrics, execution notes,
and frozen-input hashes.

## Interpretation and limitations

Face2Face is the second of four planned manipulation holdout folds. Its held-out
video ROC-AUC is 0.843724 and balanced accuracy is 0.746429 at the frozen 0.5
threshold. Across-fold conclusions must wait for FaceSwap and NeuralTextures. This is
a single-seed FaceForensics++ LOMO experiment and does not establish cross-dataset or
identity-disjoint generalization.
