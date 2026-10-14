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


## Post-publication recovery: Multiple Proxy (2026-10-14)

The published V6 package is an immutable historical baseline. The advisor clarified that attacks operate on **intermediate image representations**; the all-proxy clean-output-correctness gate was an unnecessary data restriction for multi-proxy attacks.

**Authoritative follow-up protocol:** [MULTIPLE_PROXY_RECOVERY.md](MULTIPLE_PROXY_RECOVERY.md), [multiple_proxy_recovery.yaml](multiple_proxy_recovery.yaml).

- Keep BOTH original Multiple Proxy blocks and every listed ensemble unchanged.
- Select 30 ground-truth ImageNet images per directed transition using a frozen common manifest, disjoint from the 48/class prototype reference bank; do NOT pre-screen on any proxy or target predictions.
- Optimize using proxy image-embedding Pull/Push and equal-L1-gradient fusion exactly as V6; target models remain post-hoc evaluation only.
- Rerun every unique Multiple Proxy ensemble on the same fixed cohort, including prior completed rows (because earlier clean-valid cohorts are not directly comparable).
- Output into `outputs/a100_precommercial_v6_multi_repr_recovery/`; never overwrite `outputs/a100_precommercial_v6/` or published results.
- Implement **one** resumable recovery launcher: `experiments/2026-10-a100-precommercial-v6/run_multiple_proxy_recovery.sh`.
- Primary recovery metric: Target targeted ASR across all selected ground-truth images; report target-clean-correct conditional target ASR, per-proxy embedding margins, and legacy conditional TASR as **separate** metrics.
- Verify original 17 missing cells can now run without being excluded by the prior proxy-output intersection. Preserve errors for genuine OOM/model failures.

**Conditional next phase:** [V7_OPTIMIZATION_GATE.md](V7_OPTIMIZATION_GATE.md). Only after the completed recovery report shows inadequate performance should Codex create an isolated V7 branch to experiment with better proxy selection, adaptive/conflict-aware gradient fusion, and image-gradient/input transformations. No commercial model calls.

## Model parameter precision: strict BF16

Advisor requirement (2026-10-14): **all floating-point parameters of all MLLMs must be BF16**, irrespective of proxy versus target role or model size. Quantized INT8/NF4, FP16 and mixed FP32/FP16 parameter subsets are forbidden in the recovery/V7 campaign. Conduct per-model recursive dtype + quantization audits before compute; FP32 input gradients and FP32 post-hoc statistics are still permitted. See AGENTS.md for the precise rule and A100 80GB sequential-offload strategy. The original two-block Multiple Proxy plan has 13 unique ensembles (65 × 30-image cells), not 14. Do not overwrite published V6.
