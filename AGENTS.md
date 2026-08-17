# DeepTector project rules

- Primary research goal: generalization to unseen deepfake manipulations.
- Do not optimize only for in-dataset accuracy.
- Never introduce frame-level data leakage. Split at video level, or identity level where metadata permits.
- Evaluation methodology must remain reproducible.
- Model modules must expose embeddings.
- Keep visual encoder, temporal encoder, frequency encoder, audio encoder, fusion, and retrieval components modular.
- Do not introduce Vector DB, local LLM/VLM, diffusion preprocessing, or audio-visual modeling unless the task explicitly requests that milestone.
- Prefer simple measurable research baselines over large uncontrolled architectural changes.
- Every architectural change must be evaluable through an ablation experiment.
- Never claim uncalibrated sigmoid scores are calibrated probabilities.
- Tests and default development workflows must not download data or pretrained weights.
