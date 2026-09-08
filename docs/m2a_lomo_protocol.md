# M2A FaceForensics++ c23 LOMO protocol

## Scope

M2A-TASK-01 adds a deterministic Leave-One-Manipulation-Out (LOMO) protocol for
FaceForensics++ c23. It measures transfer to one FF++ manipulation that is absent from both
training and validation. This task creates and validates protocol manifests only; it does not
train or evaluate a model.

LOMO is not cross-dataset evaluation because every record remains within FaceForensics++. It is
not identity-disjoint because the authoritative M1 manifest has no verified identity assignment
that establishes that property. A successful LOMO result would therefore not establish universal
or fully manipulation-agnostic deepfake detection.

## Fold construction

The source is the existing combined FF++ c23 manifest. Its `train`, `validation`, and `test`
assignments are preserved exactly; no videos or source lineages are randomly re-split. The
assignments are additionally reported as verified official assignments only when the input SHA-256
matches the frozen, previously audited M1 manifest.
Four folds are emitted in this fixed order:

1. hold out `Deepfakes`
2. hold out `Face2Face`
3. hold out `FaceSwap`
4. hold out `NeuralTextures`

For each fold:

- train contains every real training video and fake training videos from the three seen methods;
- validation contains every real validation video and fake validation videos from the three seen
  methods;
- test contains every real official-test video and fake official-test videos from only the held-out
  method.

Records within each generated manifest are sorted by split (`train`, `validation`, `test`) and then
by `video_id`. Portable `${DEEPTECTOR_DATA_ROOT}` paths are retained.

## Generation and audit

With `DEEPTECTOR_DATA_ROOT` pointing to the dataset parent, run:

```powershell
deeptector-prepare-ffpp-lomo `
  --source-manifest data/manifests/ffpp_c23.csv `
  --output-dir data/manifests/ffpp_lomo_c23
```

The ignored output directory contains:

```text
data/manifests/ffpp_lomo_c23/
├── deepfakes.csv
├── face2face.csv
├── faceswap.csv
├── neuraltextures.csv
└── protocol.json
```

`protocol.json` records the protocol version, source manifest path and SHA-256, fold order, held-out
method, output manifest SHA-256, split/label/manipulation counts, and integrity results. Generation
fails before publication when it finds duplicate video IDs, missing referenced files, unknown or
inconsistent manipulation metadata, source lineage crossing source-manifest splits, a changed
source assignment, or an invalid held-out test composition. Output paths are checked so generation
cannot overwrite the source M1 manifest. Each file is first written to a temporary path and then
atomically replaced, with `protocol.json` published last. This is per-file atomicity, not an atomic
transaction across the five-file set. Future consumers must verify the selected fold hash against
`protocol.json` before training.

When the source SHA-256 equals the frozen M1 manifest hash, generation additionally asserts these
known counts rather than assuming them for another source:

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 720 | 2160 seen | 2880 |
| Validation | 140 | 420 seen | 560 |
| Held-out test | 140 | 140 held-out | 280 |

Generated manifests remain under the repository's ignored `/data/` policy. Dataset videos,
checkpoints, predictions, and run directories are not added to Git.

## Future training handoff

A later, separately approved task may create one new M2 experiment configuration per fold. The
existing config loader can point its train, validation, and test manifest fields at the same
combined fold CSV and select records by their preserved `split` value. That task must use a new M2
run directory, validate the protocol/source/fold hashes before consumption, and must not alter the
frozen M1 config, threshold, checkpoints, or evaluation artifacts. LOMO training, repeated seeds,
external cross-dataset evaluation, and model changes are outside this protocol-generation task.
