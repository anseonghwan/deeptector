# M2A LOMO FaceSwap fold result

## Status

- Fold: FaceSwap held out
- Seed: 42
- Status: complete
- Training device: Intel XPU (`xpu`)
- Best checkpoint: epoch 6 (`validation_loss=0.396936`)
- Stopping: early stopping at epoch 11 (`patience=5`)
- Held-out test policy: evaluated exactly once after validation acceptance

This fold trained on original, Deepfakes, Face2Face, and NeuralTextures videos while
reserving FaceSwap for the held-out test. Frozen M1 inputs and artifacts were not
changed.

## Evaluation results

| Split | Level | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Specificity | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Frame | 0.872246 | 0.952323 | 0.818304 | 0.894606 | 0.858929 | 0.876404 | 0.696429 | 0.777679 |
| Validation | Video | 0.908418 | 0.965314 | 0.864286 | 0.923645 | 0.892857 | 0.907990 | 0.778571 | 0.835714 |
| Held-out test | Frame | 0.870297 | 0.888119 | 0.776339 | 0.739737 | 0.852679 | 0.792202 | 0.700000 | 0.776339 |
| Held-out test | Video | 0.918776 | 0.926354 | 0.803571 | 0.763975 | 0.878571 | 0.817276 | 0.728571 | 0.803571 |

Confusion matrices use `[[TN, FP], [FN, TP]]`:

- Validation frame: `[[780, 340], [474, 2886]]` (4,480 frames)
- Validation video: `[[109, 31], [45, 375]]` (560 videos)
- Held-out test frame: `[[784, 336], [165, 955]]` (2,240 frames)
- Held-out test video: `[[102, 38], [17, 123]]` (280 videos)

## Execution observations

Training completed without a process restart. Ten AMP gradient-overflow events were
handled by scaler reduction and optimizer-step skipping. A long wall-clock pause
occurred during epoch 6 while the computer was inactive; the original process stayed
alive and completed the epoch, so checkpoint resume was not used.

Train face detection attempted 253,440 frames and failed on 1,409 (0.5560%);
validation attempted 49,280 and failed on 341 (0.6920%). Evaluation failures were
31/4,480 (0.6920%) for validation and 20/2,240 (0.8929%) for held-out test.

## Artifacts

- Run directory: `runs/m2a_lomo_faceswap_seed42`
- Validation evaluation: `evaluations/validation/20260912T000031793301Z`
- Held-out test evaluation: `evaluations/test/20260912T035855604819Z`
- Reproducibility record: `docs/m2a_faceswap_reproducibility.json`

## Interpretation and limitations

FaceSwap is the third of four planned manipulation holdout folds. Its held-out video
ROC-AUC is 0.918776 and balanced accuracy is 0.803571 at the frozen 0.5 threshold.
The NeuralTextures fold remains before across-fold reporting. This single-seed FF++
LOMO experiment does not establish cross-dataset or identity-disjoint generalization.
