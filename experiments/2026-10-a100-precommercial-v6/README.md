# A100 Pre-Commercial V6

Start here, then read `AGENTS.md` in this directory before changing code or launching jobs.

## Goal

Complete every open-source experiment in the current experimental spreadsheet before commercial-model evaluation.

This branch is based on `experiment/a100-final-v5` and must reuse its tested A100 runner, classification-free attack path, resumability, and analysis infrastructure wherever possible.

Commercial models are explicitly out of scope.

## Fixed protocol

- Hardware: NVIDIA A100 80GB
- Attack: L_inf 16/255, 50 steps, step size 1/255, momentum 1.0, random start, seed 42
- Final loss: Pull 0.75 + Push 0.25
- No CLS/classification loss during attack
- Proxy representation: deepest vision block for main experiments
- Target analysis representation: deepest vision block
- Pooling: existing V5 spatial-token mean pooling
- Main data: 5 directed class pairs x 30 images
- TASR: target success conditioned on successful proxy targeted attack
- Reference bank: 48/class, disjoint from attack candidates
- No target signal during attack generation

## Experiment order

1. Model registry / revision / layer audit
2. Dataset manifests and clean screening
3. A100 batch calibration
4. Single Proxy
5. Embedding Layer sweep
6. Multiple Proxy, using the exact two-block design in `AGENTS.md`
7. Pull/Push ablation
8. Reverse-direction runs for asymmetry
9. Joint PCA/t-SNE + variance + DeltaR + margin + CKA/RSA + correlations
10. Final open-source-only report
11. STOP before commercial models

## Primary files to reuse from V5

- `src/primary_ml_cka/experiment/attack_generation.py`
- `experiments/2026-09-a100-final-v5/src/run_a100_final.py`
- `experiments/2026-09-a100-final-v5/src/calibrate_a100_batch.py`
- `experiments/2026-09-a100-final-v5/src/analyze_joint_geometry.py`
- V5 model adapters/backends and resumable state format

Do not copy V5 results into V6 unless a V6 table explicitly requests reuse and the data cohort, model revisions, loss ratio, layer, and metric definition are identical.

## Expected deliverables

At minimum:
- machine-readable configs for every experiment family;
- complete CSV summaries;
- per-image/per-cell state sufficient to audit TASR;
- model/revision/layer audit;
- A100 batch calibration results;
- PCA/t-SNE figures;
- class variance/dispersion tables;
- asymmetry table;
- Spearman correlation matrices and heatmaps;
- final Markdown report summarizing settings, completion, failures, and results.

The exact scientific matrix and acceptance criteria are authoritative in `AGENTS.md`.
