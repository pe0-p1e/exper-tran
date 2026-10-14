# Conditional V7: proxy selection, gradient fusion and perturbation optimization

## Advisor precision requirement — strict BF16 for every MLLM (2026-10-14)

**MANDATORY for V6 recovery and conditional V7:** Every floating-point **model parameter** of every participating MLLM (Qwen3.5, InternVL3.5, Gemma 4), **whether acting as proxy, auxiliary validator, or target evaluator**, must have dtype `torch.bfloat16` after loading. Vision towers, language towers, multimodal projectors, embeddings, output heads and any other floating-point parameter are included. No mixed FP32/FP16 parameter subsets, NF4/INT4, INT8, auto-quantization, or silent half-precision fallback. Also load CLIP/SigLIP2/DINOv2 floating model parameters as BF16 for a consistent proxy campaign unless a compatibility issue is explicitly reported and the affected experiment is stopped. Integer/bool buffers are exempt; image optimization tensors, gradients, loss accumulation and post-hoc geometry may legitimately use FP32 for numerical stability and **must not be confused with model parameter precision**.

Immediately after each model load and *before its first inference or gradient step*, recursively inspect `named_parameters()` and save per-dtype floating-parameter counts, bytes, sample names, role, pinned model revision, quantization config, and device to the recovery audit. Check for quantized layers separately—looking only at `parameter.dtype` is not sufficient to rule out quantization. A mismatch is a **hard error**: correct the loader and retry; never start a scientific run with mixed-precision MLLM weights. Make sure every caller of `load_target_for_generation` requests `precision="bf16"` rather than `"auto"`, and every surrogate `AttackConfig` requests `generative_precision="bf16"`. Disable all `BitsAndBytesConfig` quantization paths in these campaigns.

**Hardware constraint:** multi-proxy ensembles containing several large BF16 models may exceed the capacity of one A100 80GB. Never resolve this by quantizing, switching dtype, reducing model size, dropping a proxy, or silently skipping cells. Use deterministic sequential proxy forward/backward gradient accumulation with BF16 CPU/GPU offload (or verified multiple GPUs), preserving the exact equal-normalized-gradient mathematical update, reference embeddings, and all 50 steps; record offload overhead. If this is genuinely impossible, mark an explicit resource blocker with measured memory, not a fake success.

**Matrix count correction:** the original two-block Multiple Proxy workbook contains **13 distinct ensemble configurations**, not 14: 9 unique from block 1 plus 4 new combinations in block 2 (its other rows duplicate block 1). Thus the fixed-cohort rerun covers **65 unique ensemble × directed-transition cells** at 30 images each = **1,950 image-evaluation rows**. Do not invent a fourteenth combination merely to meet the mistaken earlier count.


**Do not create or execute V7 before V6 recovery is completed and checked.** This document is a conditional research plan, *not* permission to tune methods to the real target. The branch `experiment/a100-precommercial-v7-adaptive-ensemble` is created **only if the performance gate is triggered**.

## Decision gate after fixed-cohort recovery

Baseline: for each of the three existing targets (Qwen3.5-27B, Gemma 4 31B-it, InternVL3.5-14B-HF), identify the strongest available **single-proxy** run on the *same frozen 150 source-image instances* using the predeclared baseline pool. The primary comparison is unconditional target targeted `ASR_all` on identical image IDs. If adding proxies achieves >=5 **absolute percentage point** improvement over the strongest single proxy on **fewer than 2 of the 3 targets**, classify the V6 recovery as **insufficient improvement** and trigger V7. Include paired bootstrap CIs, actual difference counts, and caveats; the threshold is a pragmatic engineering gate, not a statistical significance claim. An ensemble can also be considered problematic if it improves ASR but needs disproportionate surrogate gradients/runtime.

The choice to *start* V7 may use these aggregate already-frozen baseline results. **Within V7**, do not use actual target predictions, representations or gradients to choose proxy subset, optimize weights, choose hyperparameters, or update adversarial examples. Optimize proxy selection on a disjoint held-out surrogate/model set and validation cohort; test on target once after freezing methods and adversarial PNGs. Keep discovery/validation/test splits auditable.

If gate does not trigger, **do not open a new branch automatically**: report recovery performance and recommend only additional statistical validation.

## New branch and separation

When triggered, create `experiment/a100-precommercial-v7-adaptive-ensemble` from the **current fixed V6 recovery implementation commit**. Do not overwrite the original two-block spreadsheet or the V6 results. New research outputs go to `outputs/a100_precommercial_v7_adaptive_ensemble/`. Preserve the original V6 recovery equal-mean method as E0. Provide a single resumable `run_v7_adaptive_ensemble.sh` runner; no commercial models.

Keep exact five transitions x30 same ImageNet ID manifests, 48/class disjoint prototypes, eps 16/255, 50 PGD iterations, step size 1/255, momentum 1, random start seed 42, deepest-layer default, Pull/Push .75/.25 unless independently ablated, full PNG audit. Fix gradient/FWD budget accounting: differing extra surrogate probes/TI/DI/VI must be reported with model forward/backward calls and wall-clock time, and evaluated at matched-gradient-call budgets as a secondary control when feasible.

## Stage A — Diagnose before optimization

From the completed V6 recovery states, for each proxy pair and step:
- gradient cosine mean/median; fraction negative; spatial disagreement/variance;
- gradient norm and the norm of the equal-weight fused update;
- losses, proxy image-embedding target/source margin changes, and their correlation with actual target ASR (post-hoc only).
Do not claim that high CKA/RSA implies compatible gradients or causal transfer.

## Stage B — Controlled method candidates

All versions compute gradients through each proxy's OWN visual representation and OWN reference prototypes; nothing requires image-class output logits during iterative attack.

**E0 equal-mean baseline:** per-proxy per-image L1 normalize, average, same momentum/sign step as V6.

**E1 subset selection (no new optimizer):** choose 2/3 proxies from fixed candidate pool using held-out surrogate validation, e.g. complementarity of clean-image CKA/RSA (computed on identical held-out calibration images) PLUS gradient cosine conflict AND held-out transfer ASR. Use no final target results to select. Predefine few candidate sets and share their evaluation images.

**E2 AdaEA-inspired representation weighting:** replace class-logit adversarial ratio in original AdaEA with surrogate-only representation-space leave-one-proxy-out lookahead. For proxy i, form a trial one-step (signed, projected) update along its per-image gradient, score the change of **other surrogate proxies' normalized Pull/Push losses** on that trial step; average improvement to score i, softmax weights with a prespecified temperature and clamp, stop gradients through weights (avoid second-order derivatives). Keep equal-mean as ablation; include extra forward costs. This is an adaptation inspired by AdaEA, **not** the exact published classification-logit formulation.

**E3 PCGrad-inspired conflict handling:** per-image gradient projection if `g_i · g_j < 0`, with `g_i <- g_i - (g_i·g_j)/(||g_j||²+eps) g_j`; operate on L1-normalized attack-gradient vectors, deterministic order/seed, avoid overwriting comparison gradients during projection, and record whether fused direction still decreases proxy losses. PCGrad was proposed for multi-task learning; its advantage for this attack is a hypothesis, not guaranteed.

**E4 E2+E3:** combine adaptive weights and conflict reduction, to test complementarity. Do not add untracked tricks.

**E5 DI²-FGSM-inspired input diversity:** randomly resize/pad with fixed seed/probability, differentiable image transform inside surrogate forward, backpropagate to the ORIGINAL 224×224 attack canvas; preserve pixel-space epsilon after inverse/projection, run with equal-mean and separately with best fusion. Keep loss and attack steps fixed.

**E6 optional TI-FGSM/variance tuning:** TI spatial smoothing of the fused pixel gradient using documented Gaussian-kernel sizes (e.g. 3/5) OR VMI-FGSM-inspired local-neighborhood variance tuning (extra gradient evaluations). Implement one at a time and compare added compute. Do not describe smoothing as an increase in allowed noise magnitude.

Proxy augmentation: first test sensible mixes of **already pinned** InternVL, Gemma, CLIP, SigLIP2 and DINOv2 backbones. If useful, add a *small number* (e.g. 1–2) of additional open-weight vision encoders with distinctly different features; verify their authentic model IDs, access, exact revisions, token/patch extraction, numerical gradient, input preprocess and A100 memory, before experiments. Do not automatically load every model or choose proxies by observed actual target ASR.

## Ablation matrix and go/no-go

Begin with one Qwen27B target and one second existing V6 target; do not claim a universal result from one target. Use exactly matched frozen cohorts, then extend to all three targets only if pilot verifies benefit. Compare E0, E1, E2, E3, E4, E5 and optional E6 with:
- `target_ASR_all` (numerator / 150, target outputs only *after* attack);
- clean-correct-conditioned target ASR and clean target accuracy;
- surrogate representation margin improvements and per-proxy conflict statistics;
- compute (gradient/backward/forward calls, walltime, peak GPU memory);
- paired bootstrap CIs for target ASR differences on same images, plus sensitivity analysis by class transition;
- at least three reproducibility seeds on selected finalists after initial pilot; do not repeatedly use target results to hyperparameter-search.

Stop/challenge the candidate if it increases computational cost markedly with no consistent, credible gain. Record failure results and ablation counterexamples; never fabricate an improvement.

## References / correct scope

1. Chen et al., **AdaEA**, ICCV 2023. Dynamic model weighting and disparity-reduced filtering were developed for classifier-ensemble adversarial attacks; the above `E2` uses an expressly different, **embedding-space surrogate-only** score. https://openaccess.thecvf.com/content/ICCV2023/html/Chen_An_Adaptive_Model_Ensemble_Adversarial_Attack_for_Boosting_Adversarial_Transferability_ICCV_2023_paper.html
2. Yu et al., **PCGrad**, NeurIPS 2020. Projects conflicting gradients for multi-task learning, not directly validated for this VLM targeted attack. https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
3. Xie et al., **DI²-FGSM**, CVPR 2019. Differentiable input transformations for transferable adversarial attacks. https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Improving_Transferability_of_Adversarial_Examples_With_Input_Diversity_CVPR_2019_paper.html
4. Wang & He, **Variance Tuning**, CVPR 2021. Neighborhood gradient-variance adjustments for transferability, with extra computation. https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.html
