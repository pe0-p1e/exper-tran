# V7 first campaign: redo every low/undefined row in the original Excel

**Authority:** user request 2026-10-19 to repeat every weak experiment in the original spreadsheet, with improved attack methods, **before** broadening the project to unrelated research or adding commercial models. Keep original workbook layout, rows, and two Multiple Proxy blocks unchanged. Save V7 data to a separate output root and only fill existing result cells if an Excel update is explicitly requested. V6 history is read-only.

## 0. Metric and denominator are part of the task

Older spreadsheet displays **conditional TASR** (target success conditioned on proxy classifiers' successful targeting), often with different denominators, and `N/A` at zero denominators. Latest strict-BF16 Multiple Proxy recovery instead has **unconditional target targeted ASR** on a fixed, ground-truth-only cohort: 13 unique ensembles, 65 complete transition cells, best pooled target rates Qwen27B 13.33%, InternVL14B 14%, Gemma31B 1.33%. These percentages **cannot be compared numerically as the same metric**.

For every new cell export **both**:
- primary `target_ASR_all = target-class hits / 30` on exactly the same 30 frozen ImageNet ground-truth images per direction, ignoring proxy and target clean classifications for eligibility;
- secondary `target_ASR_clean_correct`, target clean accuracy, proxy embedding target margin, and optional well-defined legacy conditional TASR with numerator and denominator.

Success goal is `>=15/30` per target × direction, not a pooled percentage; for "above 50%" use `>=16/30`. All 5 directed pairs are retained: T01 goldfish→monarch butterfly; T03 pineapple→acoustic guitar; T05 laptop→espresso; T07 volcano→rocking chair; T09 soccer ball→school bus. Fixed same 30 image IDs, 48 reference images/class disjoint, no clean-output gating.

**No promises of 50%; if a weak original configuration remains weak, report it.** The original layer and Pull/Push arms are scientific controls; cherry-picking only improved results would invalidate the comparisons. Strict BF16 all model floating parameter weights / no quantization / A100-compatible CPU-GPU offload, surrogate-image-intermediate-feature attacks only, and target evaluation strictly post-freeze remain mandatory. Predeclare algorithms on disjoint surrogate validation to avoid target feedback/hyperparameter leakage.

## 1. Single Proxy — 12 original rows, 10 weak, rerun both strong controls

All 6 cross-family rows are low and P0:
1. Qwen3.5-2B → InternVL3.5-4B-HF, deepest visual layer 23; historical TASR 10.32%.
2. Qwen3.5-2B → Gemma 4 E4B-it, L23; 4.00%.
3. InternVL3.5-2B-HF → Qwen3.5-4B, L23; 0.67%.
4. InternVL3.5-2B-HF → Gemma 4 E4B-it, L23; 2.67%.
5. Gemma 4 E2B-it → Qwen3.5-4B, L15; 2.67%.
6. Gemma 4 E2B-it → InternVL3.5-4B-HF, L15; 10.00%.

Intra-family: 4 weak (P0/P1), 2 former high controls:
7. Gemma E2B → Gemma E4B, L15; 100.00% **CONTROL, audit possible near-identical visual towers/proxy-target leakage or representation redundancy, and rerun strict BF16 for comparability**.
8. Gemma E2B → Gemma 26B-A4B, L15; **4.00% P0**.
9. Qwen2B → Qwen4B, L23; 73.60% **CONTROL** (same new cohort, strict BF16).
10. Qwen2B → Qwen9B, L23; **16.00% P0**.
11. InternVL2B → InternVL4B, L23; **35.57% P1**.
12. InternVL2B → InternVL8B, L23; **20.81% P0**.

Every row is 5 transitions×30 fixed images. Run at least original 50-step BF16 method once as a matched control (reuse only if manifest hash, model revision, tap, budget, precision, and evaluation semantics match), then optimized finalist. Separate within-family and cross-family success claims.

## 2. Various Embedding Layers — 6 proxy-target pairs × 5 depths = 30 rows

Keep original spreadsheet's five levels; do not replace with automatically selected target-optimized depth.

| Pair | Original layers (1%,25%,50%,75%,100%) TASR | Weak/undefined rows needing optimization | Old best |
|---|---|---|---|
| Qwen2B→Gemma E4B | NA, NA, 5.36, 4.05, 4.00 | all 5 | L11, 5.36% |
| InternVL2B→Qwen4B | NA, NA, 2.65, 1.34, 0.67 | all 5 | L11, 2.65% |
| Gemma E2B→InternVL8B | NA, NA, NA, 16.19, 4.67 | all 5 | L11, 16.19% |
| Qwen2B→Qwen4B | NA, NA, 91.96, 85.14, 72.80 | first 2; remaining 3 controls | L11, 91.96% |
| InternVL2B→InternVL4B | NA, NA, 33.02, 44.30, 38.67 | all 5 | L17, 44.30% |
| Gemma E2B→Gemma E4B | NA, NA, 0.00, 95.19, 100.00 | first 3; remaining 2 controls | L15, 100.00% |

Total **25 weak/undefined layer settings** and **5 historical high controls**. The 1%/25% NA rows generally had zero proxy-targeted-success denominator, **not evidence that the actual target was unattackable**. Implement multi-layer global token representation as a new **additional method arm**, not an alteration to the defined one-layer sweep. Validate actual model vision depth, masked token count, early-layer gradient path; do NOT silently substitute deep features for shallow-layer cells. For all 30 layer settings, rerun unchanged BF16 baseline on the same cohort before scientific comparisons; optimizer candidates and layer weights are calibrated on held-out surrogates.

## 3. Multiple Proxy — all 13 unique ensembles weak under new primary ASR

New strict BF16 fixed-cohort historical rates; 5 transitions×30 images per ensemble. The V6 recovery baseline is already scientifically suitable to reuse if code/manifests unchanged. **Rerun all 13 with improved loss/fusion**; do not repeat identical E01-E13 baseline attacks merely to consume GPU.

Target Qwen27B:
- E01 InternVL2B: 0%.
- E02 InternVL2B+4B: 1.33%.
- E03 InternVL2B+4B+8B: 3.33%.
- E10 InternVL2B+Gemma E2B: 2.67%.
- E11 E10+CLIP ViT-L/14: 6.67%.
- E12 E11+SigLIP2: 13.33% (best; **12/30 T01, 0/30 T03, 0/30 T05, 7/30 T07, 1/30 T09**).
- E13 E11+DINOv2: 5.33%.

Target Gemma31B:
- E04 Qwen2B: 0%.
- E05 Qwen2B+4B: 0.67%.
- E06 Qwen2B+4B+9B: 1.33% (severe across all five directions).

Target InternVL14B:
- E07 Gemma E2B: 5.33%.
- E08 Gemma E2B+E4B: 8.67% (**audit gradient cosine ~0.999956: nearly redundant vision signal**).
- E09 Gemma E2B+E4B+26B-A4B: 14.00% (**7/30 T01, 0 T03, 1 T05, 13 T07, 0 T09**).

Keep the workbook's two original blocks and duplicate display rows; run duplicate ensemble conditions only once when exact settings/cohort match. Order re-optimization: (1) best/representative heterogeneous E12 vs E13 to test embedding loss and gradient conflict; (2) harsh Qwen→Gemma E06 and Gemma→InternVL E09; (3) generalize validated finalist to all E01–E13. Make matched single-proxy experiments for every proxy inside the final ensembles, including CLIP/SigLIP2/DINO, to avoid claiming an ensemble win against an arbitrarily weak singleton.

## 4. Ablation Study — keep 5 unique Pull/Push ratios × 2 pair targets

Gemma E2B→Gemma E4B and Qwen2B→Qwen4B; weights:
1. 1.00/0.00, historical pooled conditional TASR 92.86%.
2. 0.25/0.75, 66.67% but one model/direction subgroup has very small denominator.
3. 0.50/0.50, 83.64%.
4. 0.75/0.25, 87.32%, **same as original main loss control**.
5. 0.00/1.00, 33.33%, with 0/1 and other zero proxy-success denominators in some directions.

The sixth visual row in Excel repeats 0.75/0.25 and **must reuse** exactly the fourth measurement, never duplicate-run.

Run both pairs × five ratios × five directions (50 unique planned cells). Do not pool conditional TASR across model pairs without reporting individual denominators; report unconditional target ASR for every pair/ratio/direction. All five ratios should be compared on the same image cohort and strictly BF16. Because source-only push cannot aim at a specific target class, low push-only targeted ASR is an expected *negative control*, not a flaw to conceal. We should NOT optimize every ratio until it exceeds 50%, as that would invalidate a controlled ablation of the loss mechanism. Similarly preserve weaker layers as interpretable negative controls.

## 5. Execution order and ablation budgets

**P0 Diagnostics + common infrastructure**: (i) audit target label parser offset 0–9 vs internal 1–10; clean target output histograms for T03/T05; (ii) all-float-weights BF16, model revision, attack/ref image disjointness, nonzero gradient for each visual tap, output reference precision, numeric clipping and frozen PNG epsilon; (iii) create fixed manifests for SINGLE, LAYERS, ABLATION matching V6 recovery, without any clean-screen filter. Establish error bars and per-cell 30-image counts.

**P1 Surrogate-development pilots (NOT actual target cherry-picking)**: shortlist A0 original pull/push, A1 all-class hard-negative margin, A2 multimodal reference/prototype distribution, A3 multi-layer features, A4 patch/local+global. Use 2-image smoke then 30-image surrogate-dev representative class directions **including T03/T05**. Select only a few promising arms from independent surrogate validation. Record forward/backward calls and gradient/Loss plots. Current component code is in `src/representation_objectives.py` and `src/attack_optimizers.py`, but currently needs integration and tests.

**P2 Freeze and run new Single Proxy controlled rows** (10 weak + 2 controls; all five directions). At the same time collect full layer sweep (25 weak + 5 controls). Complete strongest single proxies for Qwen27B, Gemma31B, InternVL14B on **the same cohort** as the future ensemble comparisons.

**P3 Gradient-fusion/attack-search pilots**: for E12 vs E13, E09, E06, compare original equal mean, adaptive cross-surrogate weights, PCGrad/conflict projection, and with/without input diversity; add 100/200-step, multi-start only in separately labeled compute-budget arms. No actual target-feedback-driven method selection. Freeze best surrogate-validated method.

**P4 Complete all 13 Multiple Proxy combos**, five directions each, using the frozen V7 algorithm. Reuse V6 strict BF16 equal-mean baseline where hashing matches. Independently finish 2 pairs × 5 Pull/Push ablation ratios, five directions each, with the same original base loss definition; other new methods are separate `loss_design_ablation`, not silently merged into the weight ablation.

**P5 Integrity/analysis/report**: one row per `sheet,exp,proxy,target,layer_or_weights,transition,algorithm,budget,seed`, exact numerator/denominator, target clean-correct subset, surrogate target margin, CKA/RSA/DeltaR as posthoc, runtime, GPU memory, model dtypes, image/manifests hash, target-parser status, source code SHA, and failure. Produce a per-sheet migration file with original row labels and exact new result mapping; NEVER modify workbook formatting. Mark as achieved **only** when each Target's five fixed directions have >=15/30 (prefer 16/30) using the same predeclared final algorithm. Failures remain reported.

## 6. Workload accounting

The *displayed original matrix* contains 12 Single rows×5 directions=60 cells; 6 layer pairs×5 depths×5 directions=150; 13 unique Multiple ensembles×5 directions=65; 2 ablation pairs×5 ratios×5 directions=50. **325 cell appearances, each 30 images = 9,750 image-level evaluations before hash-based duplicate reuse**, not 9,750 independent image IDs. Identical attacks at overlapping Single/Layer/Ablation settings can be cached and reused ONLY when all semantics match; V6 strict BF16 Multiple equal-mean baseline has 65 complete cells already. Do NOT run the entire 325-cell matrix for every speculative candidate. At the main stage, ONE new frozen algorithm per cell plus deliberately selected loss/fusion ablations.

### Output naming and status

- New root `outputs/a100_precommercial_v7_targeted50/rerun_original_matrix/` with subdirs `single_proxy/`, `layer_sweep/`, `multiple_proxy/`, `ablation/`, `audits/`, `reports/`.
- Keep every source workbook tab, table and row intact. Report new ASR in a separate result file before any user-requested Excel update, and never replace legacy TASR with ASR without relabeling.
- One resumable launcher planned: `experiments/2026-10-a100-precommercial-v7-targeted-50/run_all_v7.sh` supporting `--preflight`, `--smoke`, `--pilot`, `--full`, `--resume`. This script does not yet exist; Codex must implement and test it.
- Do not mark success before actually running A100 and checking generated files.
