# M2A LOMO Deepfakes fold result

## Status

- Fold: Deepfakes held out
- Seed: 42
- Status: complete
- Training device: Intel XPU (`xpu`)
- Best checkpoint: epoch 6 (`validation_loss=0.407958`)
- Stopping: early stopping at epoch 11 (`patience=5`)
- Held-out test policy: evaluated exactly once after the validation result was accepted

This fold trained on the remaining FaceForensics++ manipulation domains and reserved
Deepfakes for the final held-out test. The frozen M1 configuration and artifacts were
not changed.

## Evaluation results

| Split | Level | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Specificity | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Frame | 0.861735 | 0.948583 | 0.807143 | 0.886857 | 0.851488 | 0.868813 | 0.674107 | 0.762798 |
| Validation | Video | 0.901905 | 0.963091 | 0.848214 | 0.901679 | 0.895238 | 0.898447 | 0.707143 | 0.801190 |
| Held-out test | Frame | 0.916193 | 0.929197 | 0.803125 | 0.748354 | 0.913393 | 0.822678 | 0.692857 | 0.803125 |
| Held-out test | Video | 0.950102 | 0.956476 | 0.835714 | 0.773256 | 0.950000 | 0.852564 | 0.721429 | 0.835714 |

Confusion matrices use `[[TN, FP], [FN, TP]]`:

- Validation frame: `[[755, 365], [499, 2861]]` (4,480 frames)
- Validation video: `[[99, 41], [44, 376]]` (560 videos)
- Held-out test frame: `[[776, 344], [97, 1023]]` (2,240 frames)
- Held-out test video: `[[101, 39], [7, 133]]` (280 videos)

Specificity and balanced accuracy were derived from the stored confusion matrices.
The evaluation code used for this run did not yet serialize those two fields. The
metrics implementation now records them directly for subsequent folds; the Deepfakes
held-out test was intentionally not rerun.

## Training observations

The first epoch-6 attempt stopped on a transient OpenCV/FFmpeg frame-read failure for
`781.mp4` at frame 270. The frame was readable on a follow-up check, and training was
resumed deterministically from the epoch-5 `last_checkpoint.pt`. Epochs 6 through 11
then completed. AMP gradient overflow occurred 11 times across the combined logs;
each event reduced the scaler and skipped only the affected optimizer step, as
designed.

For the resumed segment, train face detection attempted 138,240 frames and failed on
702 (0.5078%). Validation attempted 26,880 and failed on 168 (0.6250%). Evaluation
face-detection failures were 28/4,480 (0.6250%) for validation and 29/2,240 (1.2946%)
for held-out test.

## Artifacts

- Run directory: `runs/m2a_lomo_deepfakes_seed42`
- Validation evaluation: `evaluations/validation/20260910T001521190904Z`
- Held-out test evaluation: `evaluations/test/20260910T053251280969Z`
- Reproducibility record: `docs/m2a_deepfakes_reproducibility.json`

The machine-readable record contains the exact artifact hashes, metrics, interruption
history, and frozen-input hashes.

## Interpretation and limitations

The held-out Deepfakes result clears the fold execution path and establishes one LOMO
result, but it is not the final M2A conclusion. Three additional manipulation holdout
folds must finish before aggregate reporting. This is a single-seed FaceForensics++
LOMO experiment; it does not establish cross-dataset generalization or identity-
disjoint performance.
