# V6 recovery protocol — Multiple Proxy, representation-only eligibility

## Advisor precision requirement — strict BF16 for every MLLM (2026-10-14)

**MANDATORY for V6 recovery and conditional V7:** Every floating-point **model parameter** of every participating MLLM (Qwen3.5, InternVL3.5, Gemma 4), **whether acting as proxy, auxiliary validator, or target evaluator**, must have dtype `torch.bfloat16` after loading. Vision towers, language towers, multimodal projectors, embeddings, output heads and any other floating-point parameter are included. No mixed FP32/FP16 parameter subsets, NF4/INT4, INT8, auto-quantization, or silent half-precision fallback. Also load CLIP/SigLIP2/DINOv2 floating model parameters as BF16 for a consistent proxy campaign unless a compatibility issue is explicitly reported and the affected experiment is stopped. Integer/bool buffers are exempt; image optimization tensors, gradients, loss accumulation and post-hoc geometry may legitimately use FP32 for numerical stability and **must not be confused with model parameter precision**.

Immediately after each model load and *before its first inference or gradient step*, recursively inspect `named_parameters()` and save per-dtype floating-parameter counts, bytes, sample names, role, pinned model revision, quantization config, and device to the recovery audit. Check for quantized layers separately—looking only at `parameter.dtype` is not sufficient to rule out quantization. A mismatch is a **hard error**: correct the loader and retry; never start a scientific run with mixed-precision MLLM weights. Make sure every caller of `load_target_for_generation` requests `precision="bf16"` rather than `"auto"`, and every surrogate `AttackConfig` requests `generative_precision="bf16"`. Disable all `BitsAndBytesConfig` quantization paths in these campaigns.

**Hardware constraint:** multi-proxy ensembles containing several large BF16 models may exceed the capacity of one A100 80GB. Never resolve this by quantizing, switching dtype, reducing model size, dropping a proxy, or silently skipping cells. Use deterministic sequential proxy forward/backward gradient accumulation with BF16 CPU/GPU offload (or verified multiple GPUs), preserving the exact equal-normalized-gradient mathematical update, reference embeddings, and all 50 steps; record offload overhead. If this is genuinely impossible, mark an explicit resource blocker with measured memory, not a fake success.

**Matrix count correction:** the original two-block Multiple Proxy workbook contains **13 distinct ensemble configurations**, not 14: 9 unique from block 1 plus 4 new combinations in block 2 (its other rows duplicate block 1). Thus the fixed-cohort rerun covers **65 unique ensemble × directed-transition cells** at 30 images each = **1,950 image-evaluation rows**. Do not invent a fourteenth combination merely to meet the mistaken earlier count.


**Status:** authoritative follow-up to the 2026-10-09 published V6 package. This file overrides only the pre-attack clean-output gating and reporting of the *new Multiple Proxy recovery*, not historical V6 methods/results, the other experiment families, or the original two-block ensemble matrix.

## Motivation and scientific correction

The advisor clarified that the proposed attack manipulates intermediate image representations, not proxy model output labels. The V6 `clean_screen.py` accidentally made all models' final-class predictions a **prerequisite** to generate an embedding-space attack:

```python
valid = [r for r in candidates
         if all(predictions_m[r.image_id] == source for m in proxies + [target])]
```

That filtering caused 17 missing transition cells: the 3 heterogeneous InternVL+Gemma+CLIP(+SigLIP2/DINOv2) ensemble rows (5 directions each) and 2 directions of the Gemma three-proxy ensemble. In missing rows the intersection had only 0–2 images, versus 30 required. This is *not* evidence that the proxy embeddings were invalid or that the attack returned 0% success.

The fix is **not** to fake missing cells, silently change the original published results, or treat every proxy as a text-generating classifier. In V6, CLIP/SigLIP2 label diagnostics arise from image–text similarity; DINOv2 from a reference-prototype nearest-center classifier; those label adapters are auxiliary evaluation instruments, not a prerequisite for image-representation optimization. Generative proxy classifiers similarly must not screen the recovery's candidate images.

## Required fixed-cohort data and clean behavior

1. Use only ImageNet ground-truth source labels, canonical image IDs, frozen seed 42, and deterministic sort/hash to choose exactly 30 images for each of the five forward transition pairs (T01/T03/T05/T07/T09); use the *same 30 image IDs* for **every** ensemble configuration within each direction. Reserve 48 reference images per class, disjoint from all attack-image IDs. Existing reference embeddings may be read if their revision+preprocessing+layer+image hashes match; never contaminate prototypes with attack images.
2. Do **not** call any proxy, CLIP, SigLIP2, DINOv2 or target model's output prediction to decide whether an image may be attacked. All 30 ground-truth images enter the attack, irrespective of their clean predicted classes. A clean target misclassification is *recorded*, not silently dropped.
3. This is a **new evaluation cohort**. Previous V6 single/multiple TASRs used model-dependent clean-valid samples. They are historical references only. For matched performance, rerun all **13 unique** Multiple Proxy ensemble configurations from `experiment_matrix.yaml` (3 Qwen-target scaling + 3 Gemma-target scaling + 3 InternVL-target scaling + 5 Qwen-target diversity = 14, with duplicate rows reused for the original two workbook blocks), five transitions each. That is 65 unique ensemble–transition cells at 30 images each; do not run duplicate spreadsheet rows twice. The original two-block table presentation remains unchanged.
4. If the 30 ground-truth images are not actually available or disjoint, stop with a *dataset-level* error and explain the missing files; do not resample based on model performance.

## Keep attack semantics unchanged

For this V6 recovery, **do not optimize algorithm choice yet**. The controlled objective is to isolate the clean-screening effect.

- Every proxy's differentiable image embedding is extracted at the deepest actual visual block, V5 spatial token mean pooling, and compared with the **same proxy's** source and target reference prototypes.
- Loss per proxy: `0.75*(1-cos(z_adv,mu_target)) + 0.25*(1+cos(z_adv,mu_source))`; no language/classification/CLS loss in the iterative optimization.
- Compute independent per-image `dL_i/dx`, L1-normalize each, take their equal-weight mean, update with `momentum=1.0`, negative sign step `1/255`, 50 steps, `L_inf <=16/255`, random start, seed 42, image 224x224, BF16 A100-compatible.
- Do not concatenate heterogeneous embeddings or ever use target outputs, target gradients, target model weights, target prototype geometry, or target ASR to construct adversarial images, weights, or proxy combinations.
- After the adversarial PNGs are saved and frozen, perform target inference and post-hoc analysis; no feedback from evaluation into attack.
- For CLIP/SigLIP2, use the internal visual hidden states for attack. Text–image class similarity is permitted **only for reported classification diagnostics**. For DINOv2 use visual hidden states; nearest-reference-prototype classification is permitted **only for diagnostics**. These are not native generated class outputs.

## Metrics: do not conflate embedding manipulation and output prediction

**Primary recovery metric**: `target_ASR_all = (# images whose frozen adversarial image makes target output the intended target label) / 30` for each cell. Ground-truth-selected samples are in the denominator regardless of target clean prediction. Explicitly label this target targeted attack success rate, **not TASR**.

Also record, at minimum:
- `target_clean_correct_count / 30`;
- `target_ASR_clean_correct = target_targeted_hits_on_target_clean_correct / target_clean_correct_count` (NA when denominator 0);
- target output/parser failures as their own status, and both conservative as-failure and parsed-only sensitivity analysis; never replace model outputs using guesses;
- target-side original-dimensional embedding changes (DeltaPull, DeltaPush, DeltaR, clean/adv margin) post hoc only;
- per-proxy surrogate embedding movement toward its target/source prototype (loss and cosine-margin improvement), for ALL 30 images; this is the direct output-independent attack objective;
- optional per-proxy targeted class-label hit masks computed post hoc via documented distinct classification adapters; all-proxy/any-proxy/majority hit rates as secondary diagnostics only;
- **legacy conditional TASR** separately, `P(target_hit | all_proxy_classifier_hit)`, only if all proxy classification adapters were explicitly defined and the all-proxy-hit denominator >0. Otherwise NA; no class-prediction gating to get those masks, and never identify this with the primary ASR.
- identical image IDs and numerator/denominator for every target/ensemble comparison, manifest and image hashes, seed, exact revisions/taps, loss settings, runtime, peak VRAM and failure counts.

When comparing two ensembles, use paired image-level differences in `target_ASR_all` on the **exact same target/image set**. Conditioned TASR with changing denominators is *not* the basis for ranking ensembles. Aggregate successes/counts across five transitions rather than taking an unweighted average of percentages with unequal denominators.

## Implementation guidance: actual code to fix

- `src/clean_screen.py` is currently doing all-model output intersection; implement a separate **ground-truth manifest construction path** for this recovery instead of executing that gate.
- `src/run_multi_proxy.py` accepts a cohort file and currently rejects anything with fewer than 30 records. Point it at the new fixed reference-independent cohort, preserving its original per-proxy gradient fusion and target isolation.
- `src/evaluate_multi_proxy.py` and `src/analyze_campaign.py` currently hardcode `outputs/a100_precommercial_v6`. Refactor to accept explicit `--output-dir`/paths and ensure recovery outputs cannot overwrite old V6 results.
- `src/campaign.py` and `run_all.sh` contain old stage completion markers. **Do not** simply rerun old `run_all.sh`, which might skip the corrected screening and experiments or overwrite previously published files. Provide an isolated launcher `experiments/2026-10-a100-precommercial-v6/run_multiple_proxy_recovery.sh` that checks prerequisites, constructs/checks fixed manifests, smoke-tests CLIP/DINO+VLM heterogeneous attacks with 2 images, runs all unique combinations, evaluates frozen images, aggregates, validates, writes a report and exits. This is ONE resumable command.
- Existing successful published attack artifacts are not interchangeable with new fixed-cohort artifacts. Never count old result cells as new completions. Use atomic write/verify before resume.
- New output root `outputs/a100_precommercial_v6_multi_repr_recovery/`. Leave `outputs/a100_precommercial_v6/`, especially `published/`, **completely untouched**.

## Acceptance checklist

- 13 unique configurations × 5 directions × 30 source-ground-truth images = **65 unique cells and 1,950 sample evaluations** (some images reused across ensembles; total rows count, not number of distinct image IDs).
- The 17 previously skipped cells now undergo an actual optimization attempt and target evaluation; no `insufficient_clean_valid` on the basis of any proxy/target prediction. If A100 OOM or model error occurs, record a genuine execution failure and continue others.
- Every cell uses same 30 image IDs for its direction, matching SHA256 of the selected image manifest.
- Exact noise bound audited pixelwise on final saved PNG `max |adv-clean| <=16/255` and 50 steps, 0.75/0.25, BF16 where supported.
- For primary ASR denominator zero cannot occur; report exact target-hit numerator over 30, even if 0 success. Secondary diagnostic denominators of zero are NA.
- Compare 1-, 2-, 3- and 4-proxy ensembles against strong single-proxy baselines **on the newly fixed cohort**, with image-level paired bootstrap confidence intervals, seeds logged; use a descriptive caveat for target clean errors and output parse errors.
- Emit `reports/multiple_proxy_recovery_report.md`, `reports/multiple_proxy_recovery_completion.csv`, `reports/paired_ensemble_comparison.csv`, `audits/recovery_validation.json` and all per-image results/checkpoints.
- Smoke test no class-gating and resume-after-interruption; all work in the new output root; commit code/configs/metadata and a linkable **compact summary** (do not indiscriminately commit multi-gigabyte model caches).

## Controlled sequence and conditional research branch

1. Fix cohort generation and output-root handling; unit/smoke tests.
2. Run the entire existing multi-proxy matrix unchanged, with a single recovery script.
3. Analyze target ASR, per-proxy geometric movement, gradient alignment, per-target best single baseline, and failures.
4. Apply `V7_OPTIMIZATION_GATE.md`. Only if the documented performance trigger is met, **create a separate branch** (never silently alter this baseline) and implement literature-inspired proxy selection, adaptive/gradient-conflict fusion and input-gradient/noise optimization.
5. Never evaluate commercial models without an explicit new request.
