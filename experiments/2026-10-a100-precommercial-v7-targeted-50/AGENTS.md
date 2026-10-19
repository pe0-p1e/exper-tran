# V7 targeted transfer optimization — 50% per-target / per-direction objective

## Status and overriding user instruction

The user has explicitly overridden the earlier V6 recovery's `do_not_trigger_v7` gate: the observed max 13.3–14.0% pooled target ASR is far below the desired standard. This V7 is authorized **now**, independent of the old gate. Research objective: **>=50% targeted ASR on EACH of the three targets and EACH of the five original directed transitions**, NOT just the mean or whichever easy classes succeed. With 30 images per direction, `>=15/30` is the minimum, and `>=16/30` is the preferred strictly-above-50% milestone. For each target, all five directions must meet the minimum; selected final algorithm and evaluation protocol must be clearly identified. Report lower results truthfully; this is an aspirational performance target, not a license to fabricate success or game evaluation.

You must read these before changes:
- `ALGORITHM_REDESIGN.md` — mandatory detailed implementation of feature objective, ensemble fusion and transfer optimization, with controlled experiment ladder.
- `../2026-10-a100-precommercial-v6/MULTIPLE_PROXY_RECOVERY.md`
- `../2026-10-a100-precommercial-v6/multiple_proxy_recovery.yaml`
- `../2026-10-a100-precommercial-v6/V7_OPTIMIZATION_GATE.md`
- `README.md` and `experiment_plan.yaml` in this V7 directory
- V6 recovery per-cell summary, image-level evaluation, gradient compatibility, BF16 audit and pinned revisions.

### Scientific invariants

- Open-source only. No commercial APIs.
- All floating-point MLLM weights strictly `torch.bfloat16` in EVERY role, including complete vision + language backbones and heads. Vision-only proxies also BF16. NO quantization, NF4, INT8, FP16/FP32 weight fragments. Audit `named_parameters()` and quantization state BEFORE each run. Float32 image gradients, normalization, statistics are allowed. Sequential model CPU/GPU offloading where required.
- Surrogate-only attack. Target prediction, target embedding, target logits, or target gradient must **never** select proxy models, ensemble weights, loss, hyperparameters, images, or noise during attack optimization or tuning. Held-out surrogate models and separate dev images for method selection. Actual target output ONLY after adversarial PNGs are frozen and method chosen; no repeated target-guided search.
- Frozen same 30 ImageNet source-ground-truth images per T01,T03,T05,T07,T09, same 48/class reference bank, as V6 recovery; independent of ANY clean-classification output. Do not drop hard images, switch target classes, change prompt/parser, or combine easy directions to meet 50%.
- Keep a **directly comparable primary budget**: `L_inf epsilon=16/255`, 224×224 canonical attack canvas, 50 iterations, step=1/255, momentum=1, random start, seed=42, exactly 30 images per transition; test seed 43 and 44 for finalists. Larger iteration budgets (100/200), extra restarts, and extra surrogate forwards permitted ONLY as explicitly labeled separate compute-budget ablations with full query/forward/backward counts; never mix them with 50-step results without disclosure. Epsilon must not be raised in the primary benchmark.
- Attack objective uses proxy image **intermediate representations** (global and optionally patch/local/multi-layer hidden states). The 10 source/target image feature reference banks are available. Do not require output classifiers for loss; CLIP/SigLIP2 textual features, if explored, belong in a clearly distinguished optional ablation, not silently into the core vision-only protocol.
- Freeze new attack artifacts before target evaluation and continue using fixed prompt and strict parsing. Report unconditional **target targeted ASR** per target AND direction; additionally report clean-correct ASR, source retained rate, off-target changes, per-proxy embedding margin and optimization diagnostic, not only success-only examples. Historical conditional TASR is a different metric.

### Evidence threshold and scientific reporting

- No claim of success unless independently audited target ASR `>=15/30` in **all 15 target×direction cells** on locked cohort, with exact success masks, eps audit, parser status, BF16 audit. Prefer `>=16/30` to avoid ambiguity on 'above 50%'.
- Full matched single-proxy comparisons are mandatory (not just the original single proxy per target). Re-evaluate **all candidate individual proxy models used in final ensembles** on the same cohort with identical final attack objective and comparable compute before claiming ensemble improvements.
- Do not optimize only means. Add a `minimum_target_direction_ASR` column and an `all_15_cells_above_50pct` boolean. Mark failed direction counts and attack families. Compute per-direction binomial/bootstrap uncertainty; per-cell n=30 is not precise evidence of robust generalization.
- Validate finalists over seeds 42/43/44 on the same cohort and, if feasible, independent untouched 30-image holdout per direction. Never choose seed based on actual target results.
- Keep unsuccessful trials; do not tune to target, hide failures, alter denominator, claim guaranteed gains or misrepresent other papers' results as equivalent targeted 10-class accuracy.

### Ordered experimental strategy

**Phase 0 — audit why current gradients fail.** Check label indexing (raw model replies 0–9 versus internal labels 1–10), prompt parser, target output distributions on failed T03/T05, preprocessing/normalization and native model resolution, image/reference overlap, gradient connection/backprop from raw image to deepest and intermediate vision layers, BF16 near-constant embedding artifacts especially Qwen and Gemma, gradient saturation, PGD projection and PNG delta. Verify Gemma E2B/E4B near-duplicate vision representations and cosines near 1.0. Inspect per-class proxy target-margin optimization, not only resulting target labels. Do not 'correct' benchmark by changing target parsing criteria.

**Phase 1 — strongest standalone baselines on identical images.** For EACH target and each of 5 directions, test all available SINGLE proxies from the candidate pool (Qwen2B/4B/9B, InternVL2B/4B/8B, Gemma E2B/E4B/26B, CLIP, SigLIP2, DINOv2 as A100 permits, excluding the target itself). At least start with existing E01/E04/E07 plus CLIP, SigLIP2, Gemma E2B and relevant cross-family surrogates. Use surrogate-only validation to choose a small set to run; target results are evaluation-only. Retain reference V6 recovery equal-mean E01–E13 as static baselines.

**Phase 2 — improve the representation loss.** Compare clean, tightly controlled methods: (a) existing 0.75 target pull / 0.25 source push; (b) 10-class target-vs-hardest-negative margin loss using each proxy's image class prototypes, with stable temperature; (c) multi-layer global representation aggregation from selected normalized depths; (d) global + local patch-level and multi-scale feature matching using target reference images, with differentiable patch alignment and careful avoidance of trivial background matches. No target model outputs in loss. Explore prototype variance/dispersion normalization. Record per-class surrogate attack effectiveness to distinguish optimization vs transfer failure.

**Phase 3 — fuse compatible proxy gradients.** Equal L1-gradient averaging versus held-out-surrogate-selected 2/3/4-member subsets, AdaEA-inspired **representation-specific** weights computed exclusively from surrogate loss/gradient improvement, PCGrad-inspired conflict projection, consensus/negative-gradient suppression, and adaptive+conflict combo. Include computation-matched baselines and an ablation that removes each proxy. Avoid blindly adding more proxies: V6 SigLIP2 13.3% vs DINOv2 5.3% demonstrates that identity matters.

**Phase 4 — improve input-space search.** DI2-style resize/pad/augmentation with backward mapping to original 224px tensor, multi-scale input crops, TI convolutional gradient smoothing, variance-tuned or distributional gradient aggregation, and multiple random starts on a purely **surrogate-side** loss, reporting exact extra forward/backward cost. Scale iterations 50→100→200 only as a separately labeled budget-ablation if 50-step setting fails, with epsilon unchanged and full results. Never increase epsilon to achieve the primary 50% threshold.

**Phase 5 — target held-out evaluation and synthesis.** Predeclare finalists on surrogate dev set; create/freeze PNGs; evaluate all 15 fixed target×direction cells with one original prompt and exact parsing; report full error distributions, run quantitative paired comparisons and robustness seeds. If 50% threshold is unmet, STOP and honestly explain the blocking target/direction/loss mechanism; do NOT repeatedly search against target predictions.

### Computation strategy

A100 80GB; full model BF16; sequential offload; grouped reference cache; resume per method/ensemble/transition/seed/microbatch; 2-image smoke first; test one full 30-image representative cell per algorithm before scaling; no pretense 200-step/10-model runs are free. Save optimization histories, per-step proxy loss and per-image target-margin traces; peak VRAM, walltime, forwards/backwards and cache hashes.

### Literature (method inspiration, not guaranteed results)

- Li et al., MPCAttack, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Li_Multi-Paradigm_Collaborative_Adversarial_Attack_Against_Multi-Modal_Large_Language_Models_CVPR_2026_paper.html
- Zeng et al., Everywhere Attack, AAAI 2025: https://ojs.aaai.org/index.php/AAAI/article/view/33061
- Chen et al., AdaEA, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Chen_An_Adaptive_Model_Ensemble_Adversarial_Attack_for_Boosting_Adversarial_Transferability_ICCV_2023_paper.html
- Yao et al., ensemble analysis, ICML 2025: https://proceedings.mlr.press/v267/yao25f.html
- You et al., DDGA, ICML 2026: https://proceedings.mlr.press/v306/you26e.html
- Papers involving UAP, safety bypass or untargeted ASR report a DIFFERENT attack task; do not cite their high success percentages as proof this targeted 10-class benchmark can achieve >50%.

### Required deliverables

- New branch isolated from immutable V6: `experiment/a100-precommercial-v7-targeted-50`.
- New scripts and configs under THIS V7 directory; new results under `outputs/a100_precommercial_v7_targeted50/`; preserve `outputs/a100_precommercial_v6_multi_repr_recovery`.
- `scripts/run_all_v7.sh` or an unambiguous equivalent single resumable launcher; test-only `--preflight`, `--smoke`, `--resume` supported.
- `reports/diagnostics.md`, `reports/controlled_ablation.csv`, `reports/target_direction_results.csv`, `reports/paired_baselines.csv`, `reports/final_report.md` with exact status/count; `audits/bf16.json`, `audits/linf.json`, `audits/model_revisions.json`, `audits/target_isolation.json`, `audits/runtime.json`.
- Full model details, negative results, failure reasons, successful and failed examples. Commit source and compact results, not multi-gigabyte caches.

### User execution-priority override — 2026-10-18

Read `FOCUSED_RESEARCH.md`. Pause the broad test/ablation queue. First optimize promising algorithms AND alternative multi-proxy compositions using only ten fixed development images (two per direction) and native held-out-surrogate output validation. Run small ablations only after pilot performance is promising. No changes to the 30-image formal benchmark, no target feedback during tuning, no false claim of 15/30 from a two-image pilot. Default launcher follows the focused queue; preserve existing V7 locks and artifacts.

### Approved ten-hour campaign — 2026-10-18

The user approved the ten-hour research prompt and requested actual execution. Read `TEN_HOUR_RESEARCH.md` and `ten_hour_plan.yaml`. Default launch now uses this time-budgeted research campaign, with the focused ten-image native-surrogate protocol, small loss mixtures, alternative proxy combinations, optimizer comparisons, independent method lock and untouched 30-image final evaluation. Budget exhaustion must leave honest partial reports and resumable artifacts. No final-target feedback may enter research selection.

### User-approved twenty-hour extension — 2026-10-18

The user explicitly requested additional research and a total 20-hour reservation. Read `TWENTY_HOUR_RESEARCH.md` and `twenty_hour_plan.yaml`. The original start time and completed artifacts are preserved; rerunning never adds another 20 hours. Keep final targets isolated. Add controlled layer/proxy removal, reference geometry and optimizer interactions, then nested 8-image/direction confirmation and matched singles. Existing scientific constraints remain unchanged.

### Strict family-disjoint correction — 2026-10-19

Read `FAMILY_DISJOINT_RECOVERY.md`. The user requires excluding the entire target family from attack proxies: no Qwen for a Qwen target, no InternVL for an InternVL target, and no Gemma for a Gemma target. Preserve existing valid Gemma31B cross-family results and all historical measurements. Redo Qwen/InternVL using the fixed corrected matrix. No open-ended VQA now. Freeze settings before corrected evaluation and never retune from target outcomes. Default launcher runs the corrected family-disjoint campaign in a distinct output directory.

### Commercial pilot and category-generalization planning — 2026-10-19

The user now authorizes commercial preparation and a THREE-image initial CLI connectivity pilot, overriding the earlier blanket commercial ban only for this separate study. Read `COMMERCIAL_PILOT.md`. This does not add commercial calls to the running open-weight campaign. Actual model identity, authentication, image input and isolated context must be verified; paid API routes additionally require configured endpoint/model/access/spend settings. See `HELD_OUT_CLASS_GENERALIZATION.md` for the new-category generalization proposal; it is planned, not executed, and must not interrupt or change frozen current experiments. No open-ended VQA is launched now.

### Latest ordering and paper preparation — 2026-10-19

Read `LATEST_EXPERIMENT_FRAMEWORK.md` and `latest_experiment_framework.yaml`.
The user explicitly orders: finish/report corrected experiments, then multi-proxy
selection ablation, then loss/category generalization, and commercial evaluation
last. Prepare code now without changing the running campaign's source/config
fingerprints. Submit this framework to GitHub first, then build the paper skeleton.
All unmeasured claims/results in the manuscript must be visibly pending.

### User correction: pause paper drafting — 2026-10-19

The latest user message stops paper-framework work. Do not draft or push paper
files now. Continue experiment planning/code, proxy-selection tables and the
running corrected campaign. Uncommitted paper drafts were set aside locally
under outputs/a100_precommercial_v7_targeted50/deferred_paper_scaffold/.
