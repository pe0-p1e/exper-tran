# AGENTS.md — experiment/a100-precommercial-v7-targeted-50

**Current branch-specific authoritative V7 instruction:** read `experiments/2026-10-a100-precommercial-v7-targeted-50/AGENTS.md`, `README.md`, and `experiment_plan.yaml` BEFORE editing code or launching experiments. The user explicitly requested >=50% targeted transfer ASR in EACH target × source→target-class cell; the former V6 `do_not_trigger_v7` gate is superseded. V6 remains a read-only baseline, never overwrite V6. The goals are not achieved yet. Maintain **strict BF16 model parameter weights for every model (all MLLM roles), no quantization**, fixed ground-truth cohorts, surrogates' intermediate visual representations only during attack, and target evaluation post-freeze only. Any conflict with the historical V6 instructions below is resolved in favor of the V7 AGENTS.md. Final V7 code/results must be on THIS branch, with new outputs under `outputs/a100_precommercial_v7_targeted50/`.

---

# AGENTS.md — experiment/a100-precommercial-v6

This branch exists for one task: complete all open-source experiments before commercial-model evaluation.

Before changing code or launching jobs, read:

1. `experiments/2026-10-a100-precommercial-v6/AGENTS.md` — authoritative scientific and execution instructions.
2. `experiments/2026-10-a100-precommercial-v6/experiment_matrix.yaml` — exact machine-readable experiment matrix.
3. `experiments/2026-10-a100-precommercial-v6/README.md` — execution order and deliverables.

The nested `AGENTS.md` is authoritative for the entire branch, including edits under `src/`, `configs/`, and scripts outside the experiment directory.

Critical constraints:
- do not evaluate commercial models or call commercial APIs;
- do not redesign Multiple Proxy into Phase A/B/C;
- do not change 5 source-target pairs x 30 images;
- main Pull/Push weights are 0.75/0.25;
- main proxy representation uses the deepest visual block;
- TASR is target success conditioned on successful proxy targeted attack;
- target models are evaluation-only and must not influence attack generation;
- preserve V5 outputs; write all new artifacts under `outputs/a100_precommercial_v6/`.

Finish the full matrix and analysis, produce auditable summaries/reports, then stop before commercial-model work.


## V6 Multiple Proxy recovery override (2026-10-14)

The user clarified with the advisor that proxy intermediate image representations—not model output predictions—drive the attack. The original all-proxy clean-classification screening caused 17 missing multi-proxy transition cells and is **not an eligibility requirement** for this recovery. Before implementing anything, read:
- `experiments/2026-10-a100-precommercial-v6/MULTIPLE_PROXY_RECOVERY.md`
- `experiments/2026-10-a100-precommercial-v6/multiple_proxy_recovery.yaml`
- `experiments/2026-10-a100-precommercial-v6/V7_OPTIMIZATION_GATE.md`

For the **recovery campaign only**, these override older clean-valid screening and multi-proxy primary-metric instructions wherever they conflict:
- All proxy types (VLM, CLIP, SigLIP2, DINOv2) participate through their intermediate image embeddings/prototypes. Do not require proxy clean classification or agreement before attacking.
- Select a fixed per-source-class 30-image manifest from ground-truth ImageNet labels, disjoint from reference images; selection cannot depend on *any* model outputs, including the target.
- Keep frozen V6 results and original workbook unchanged. Execute the original two-block Multiple Proxy composition matrix on the same cohort for fair comparisons and save to a distinct output root.
- Report unconditional target targeted ASR, target-clean-correct-conditioned ASR, representation changes, and diagnostic legacy conditional TASR separately. Never rename ASR as TASR or count undefined denominators as zero.
- This does not relax the ban on target feedback/gradients during attack generation.

Do **not** preemptively create the optimization branch. First complete and assess the V6 recovery. Only if the documented quality gate triggers, create the separate V7 research branch to test additional proxies, gradient fusion, and input/noise optimization algorithms.

## Advisor precision requirement — strict BF16 for every MLLM (2026-10-14)

**MANDATORY for V6 recovery and conditional V7:** Every floating-point **model parameter** of every participating MLLM (Qwen3.5, InternVL3.5, Gemma 4), **whether acting as proxy, auxiliary validator, or target evaluator**, must have dtype `torch.bfloat16` after loading. Vision towers, language towers, multimodal projectors, embeddings, output heads and any other floating-point parameter are included. No mixed FP32/FP16 parameter subsets, NF4/INT4, INT8, auto-quantization, or silent half-precision fallback. Also load CLIP/SigLIP2/DINOv2 floating model parameters as BF16 for a consistent proxy campaign unless a compatibility issue is explicitly reported and the affected experiment is stopped. Integer/bool buffers are exempt; image optimization tensors, gradients, loss accumulation and post-hoc geometry may legitimately use FP32 for numerical stability and **must not be confused with model parameter precision**.

Immediately after each model load and *before its first inference or gradient step*, recursively inspect `named_parameters()` and save per-dtype floating-parameter counts, bytes, sample names, role, pinned model revision, quantization config, and device to the recovery audit. Check for quantized layers separately—looking only at `parameter.dtype` is not sufficient to rule out quantization. A mismatch is a **hard error**: correct the loader and retry; never start a scientific run with mixed-precision MLLM weights. Make sure every caller of `load_target_for_generation` requests `precision="bf16"` rather than `"auto"`, and every surrogate `AttackConfig` requests `generative_precision="bf16"`. Disable all `BitsAndBytesConfig` quantization paths in these campaigns.

**Hardware constraint:** multi-proxy ensembles containing several large BF16 models may exceed the capacity of one A100 80GB. Never resolve this by quantizing, switching dtype, reducing model size, dropping a proxy, or silently skipping cells. Use deterministic sequential proxy forward/backward gradient accumulation with BF16 CPU/GPU offload (or verified multiple GPUs), preserving the exact equal-normalized-gradient mathematical update, reference embeddings, and all 50 steps; record offload overhead. If this is genuinely impossible, mark an explicit resource blocker with measured memory, not a fake success.

**Matrix count correction:** the original two-block Multiple Proxy workbook contains **13 distinct ensemble configurations**, not 14: 9 unique from block 1 plus 4 new combinations in block 2 (its other rows duplicate block 1). Thus the fixed-cohort rerun covers **65 unique ensemble × directed-transition cells** at 30 images each = **1,950 image-evaluation rows**. Do not invent a fourteenth combination merely to meet the mistaken earlier count.
