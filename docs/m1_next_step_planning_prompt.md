# DeepTector Milestone 1 Next-Step Planning Prompt

> Superseded: this planning prompt was consumed before the batch-8 configuration and deterministic
> epoch-boundary resume implementation were completed. Do not use it as the current launch prompt.

Copy the request below into a new ChatGPT/Codex task. For the most evidence-backed plan, attach:

1. `docs/m1_stabilization_capacity_handoff.md`
2. `runs/m1_xpu_capacity_20260828T063347Z/capacity_report.json`

If the new task has access to the local repository, attachments are optional because it can inspect
the files directly.

---

Continue planning the DeepTector graduation research project from the current LOCAL repository.

This is a PLAN-ONLY task. Inspect the current state and produce an execution-ready next-step plan
and research strategy. Do not edit files, stage, commit, push, change branches, launch training, run
full evaluation, or implement a later milestone in this task.

## Local context

Project root:
`C:\Users\user\Projects\deeptector`

Branch:
`main`

Base HEAD:
`ec4ea25 Complete FF++ M1 readiness with Intel XPU support`

XPU Python environment:
`C:\Users\user\Desktop\deeptector_\.venv`

Dataset root and environment variable:
`DEEPTECTOR_DATA_ROOT=C:\Users\user\Datasets\deeptector`

Local CLIP model:
`C:\Users\user\Datasets\deeptector\models\openai-clip-vit-base-patch16`

Read `AGENTS.md` first. Treat the current uncommitted working tree as expected project work. Do not
reset, restore, clean, delete, overwrite, or discard it. Inspect `git status`, `git diff`, staged
state, and untracked non-ignored files read-only before planning.

## Verified handoff state

The current uncommitted stabilization work adds:

- fail-fast finite guards for training logits, loss, and trainable gradients;
- AMP unscale-before-gradient-inspection behavior;
- XPU AMP initial scale `1024` after the previous default scale overflowed on the real Arc path;
- directed FF++ pair completeness checks and regressions;
- hardened artifact ignore rules;
- a bounded Intel XPU capacity benchmark, tests, and documentation.

Verification completed:

- real FF++ c23 manifest: `5000` records
- train/validation/test totals: `3600 / 700 / 700`
- duplicate IDs: none
- cross-split source overlap: none
- missing or unknown files: none
- CPU: Ruff passed; pytest `39 passed, 1 skipped`; read-only parse and diff checks passed
- XPU: torch `2.13.0+xpu`; Intel Arc available and auto-selected; pytest `40 passed`
- nothing is staged, committed, or pushed
- no dataset, model weight, checkpoint, run, manifest, or virtual-environment artifact is tracked

The final capacity artifact is:

`C:\Users\user\Projects\deeptector\runs\m1_xpu_capacity_20260828T063347Z\capacity_report.json`

Its SHA-256 is:
`6704738A75D8F3BB2D93D38013278B45881A03D03F98A4A99A68E3F27ABCC3C8`

Use this artifact, not the superseded `20260828T062717Z` report.

Measured stable frame batches:

| Batch | Stable | Train E2E fps | Compute fps | Validation fps | Peak XPU reserved | Minimum system memory available |
|---:|:---:|---:|---:|---:|---:|---:|
| 2 | 3/3 | 4.4867 | 28.1441 | 5.3260 | 386 MiB | 14.68 GiB |
| 4 | 3/3 | 4.8185 | 37.9023 | 5.6152 | 402 MiB | 14.69 GiB |
| 8 | 3/3 | 4.9634 | 50.3447 | 5.5837 | 422 MiB | 14.66 GiB |

Recommended full-training frame batch: `8`. Batch `4` is the conservative fallback. The current
`configs/experiment/ffpp_m1.yaml` still declares batch `32`; no full experiment has been launched.

At batch 8, the bounded estimate is:

- `3,600` training steps per epoch
- `700` validation steps per pass
- about `1:36:42` training per epoch
- about `0:16:43` validation per epoch
- about `1:53:25` combined per epoch
- about `37:48:28` maximum for 20 epochs, excluding checkpoint overhead and long-run variability

## Research boundary

The primary objective is generalization to unseen deepfake manipulations. Milestone 1 remains only
a reproducible frozen CLIP ViT-B/16 visual baseline. Do not propose silently changing its scientific
definition before the baseline is trained and evaluated. Do not implement or assume SBI,
frequency-domain modeling, contrastive or manipulation-invariant losses, temporal modeling,
VideoMAE, audio/fusion, diffusion, retrieval, LLM/VLM, calibration, adaptive inference, or
explanation modules as part of the immediate M1 execution.

Later-milestone ideas may appear only in a clearly separated, evidence-gated future roadmap after
the M1 baseline result is preserved.

## Planning request

Inspect the actual repository, configuration, CLI behavior, checkpoint/resume semantics, metrics,
and artifact layout. Then produce a concrete phased plan covering:

1. **Current-diff stabilization**
   - review gates for the uncommitted changes;
   - dependency-aware logical commit units and exact included files;
   - suggested commit messages;
   - checks preventing ignored/generated artifacts from entering commits.

2. **Full-run configuration decision**
   - the exact minimal configuration change needed to move from frame batch `32` to `8`;
   - whether any other existing value must change, with repository evidence;
   - a batch `4` fallback trigger without changing the scientific experiment definition.

3. **Operational preflight**
   - environment, dataset, manifest, local model, disk capacity, output path, Windows power/sleep,
     thermals, logging, and XPU health checks;
   - commands that are safe to run immediately before launch;
   - explicit pass/fail criteria and stop conditions.

4. **Full M1 launch runbook**
   - the exact proposed command, but do not run it;
   - expected duration and step counts;
   - checkpoint policy, early stopping, interruption/resume behavior, and recovery procedure based
     on what the current code actually supports;
   - if resume is not supported, identify that clearly and propose the smallest prerequisite rather
     than assuming it works;
   - monitoring cadence and the signals to record without modifying training behavior.

5. **Post-training M1 evaluation and preservation**
   - validation/test evaluation order and leakage-safe artifact handling;
   - exact existing metrics and prediction artifacts to collect, based on repository evidence;
   - reproducibility metadata, hashes, environment capture, and a concise baseline result report;
   - what should and should not be committed to Git.

6. **Research decision gate after M1**
   - criteria for judging whether the baseline is complete and trustworthy;
   - how M1 results should determine the next generalization experiment;
   - a ranked later-milestone roadmap expressed as hypotheses and ablations, not implementation
     authorization.

7. **Risk register and approval gates**
   - numerical, XPU/shared-memory, thermal, interruption, data-integrity, artifact, and research
     validity risks;
   - mitigation, detection signal, and recovery action for each;
   - explicit points requiring user approval.

## Required answer format

Return these sections:

A. Repository evidence reviewed
B. Recommended immediate objective
C. Phased execution plan with dependencies
D. Exact proposed commands, clearly marked as NOT EXECUTED
E. Acceptance criteria for every phase
F. Monitoring and recovery strategy
G. Git and artifact strategy
H. M1 evaluation strategy
I. Evidence-gated post-M1 research roadmap
J. Risk register
K. Approval gates and first proposed execution task

Separate verified facts from assumptions and recommendations. Cite concrete repository paths and
symbols. Flag discrepancies between this handoff and the current local state. Prefer the smallest
safe next action. Stop after the plan and wait for explicit approval.
