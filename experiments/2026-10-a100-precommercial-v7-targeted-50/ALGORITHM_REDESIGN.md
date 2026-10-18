# V7 algorithm redesign: surrogate-only representation attack, not proxy scaling

**This is an implementation + experiment contract.** The previously published V6 BF16 recovery's max target ASRs were Qwen27B 20/150=13.33%, InternVL14B 21/150=14%, Gemma31B 2/150=1.33%, with serious T03/T05 failures. Do not claim the algorithms below have improved those numbers before running them.

## 1. Critical baseline code observation

In V6 `src/run_recovery_ensemble.py`, the 50-step loop obtains every proxy's single deepest-layer **mean-pooled** visual representation `z` and minimizes:

`0.75*(1-cos(z,mean(ref_target))) + 0.25*(1+cos(z,mean(ref_source)))`.

Each pixel-space image gradient is normalized by per-image L1 norm, equally averaged, then accumulated with momentum and projected onto the 16/255 L-infinity ball. This means:
- Loss considers **only two class centers**, ignoring the remaining 8 distractor classes, class multimodality and hard negatives.
- **Spatial token patterns disappear under mean pooling**; target visual semantics may not transfer to a target's attended regions.
- Only deepest layer is attacked, even though the V6 layer sweep had different best layers for different proxy/target pairs.
- Some ensembles are redundant: Gemma E2B/E4B cosine of pixel gradients ~0.999956. Other hetero pairs have negative-gradient fractions 58–65%.
- The V6 metric is unconditional target ASR with fixed ground-truth cohorts; it must stay the primary metric.

Fix/determine these experimentally **before** escalating steps or adding proxy models. During audit: confirm raw target class codes 0–9 map consistently to internal class IDs 1–10; separately inspect target output confusion on all 30 failed T03/T05 samples. Check that generation parser cannot fake target hits; do not change the prompt.

## 2. Core algorithm modules already added (do not merely rewrite)

- `src/representation_objectives.py`: `original_pull_push_loss`, `class_scores` (`centroid` vs `soft_reference`), `all_class_margin_loss`, `local_patch_target_loss`, `combine_multi_layer`.
- `src/attack_optimizers.py`: `normalize_pixel_gradients`, `fuse_proxy_gradients` (equal, weighted, pcgrad, weighted_pcgrad, consensus), `adaptive_weights_from_surrogate_lookahead`, `differentiable_resize_pad`.
- `tests/test_method_components.py`: CPU unit tests for finite input gradients, loss shapes, BF16-compatible float32 computations, mask handling, selected gradients, and input diversity differentiability.

These modules are **not wired to the attack runner yet**, and have **not been executed** on the A100. Start with the tests. Maintain a controlled interface and move clean components into the canonical package only if scientifically appropriate.

## 3. Loss redesign, mathematical specification and ablations

Every loss is calculated *independently in each surrogate's own visual feature space*, with no target-model access or label generation. Reference features come from the same 48/class image banks.

### L0: original Pull/Push (reproduction)

`L0 = 0.75*(1-cos(z,mu_t)) + 0.25*(1+cos(z,mu_s))`.

### L1: full 10-class discriminative prototype margin

For each source proxy, normalize its features and class reference centers `mu_c`, and calculate `s_c=cos(z,mu_c)`. Define:
`L_margin=temperature * softplus((temperature*logsumexp({s_c/temperature:c!=t}) - s_t + m)/temperature)`
where `t` is the desired target class, `m` margin and `temperature` are configured. Use `hardest` rival as a separate ablation. It is a **surrogate feature margin**, not a target-model logit or VLM final prediction. Tune temperature and margin using disjoint surrogate-dev images only. Test L1 alone and L0+L1, with explicit weight normalization.

Why: unlike source-only push, encourages target to exceed **all** other 9 Imagenet classes. Inspect gradient magnitudes and target-vs-hardest-rival margin.

### L2: multi-positive reference distributions instead of one center

A class has 48 reference images. Rather than collapsing them to a single potentially diluted prototype, calculate
`s_c(z)=tau*logmeanexp_j(cos(z,ref_cj)/tau)`.
This approximates affinity to multiple possible target modes. Compare prototype centroid versus `soft_reference` inside exactly the same L1 margin objective. Since top-mode selection can overfit a reference instance, also evaluate class balance and held-out references. Distinguish this from stochastic target-reference image selection; both are valid named ablations.

### L3: multi-layer visual attack

Use a frozen set of normalized visual layers (e.g. 25%,50%,75%,100%). For each layer `l`, extract surrogate `z_l`, cache the 48/class reference embeddings for that exact layer, compute a valid L0/L1/L2 per-layer objective, and combine `L_multi=sum_l w_l L_l`. Select layers/weights *only* on held-out surrogates. Do not mix feature vectors with incompatible dimensions or compare model layers naively by raw index.

Ablate exactly one-layer at a time versus two layers versus four layers; report additional activation GPU memory and forward/backward budget. Verify all layer taps are genuine outputs and the visual gradient reaches the original 224px attack canvas.

### L4: target-class local patch semantics

For each proxy and layer, extract **spatial patch tokens excluding CLS and padding**. Build source and target patch-center banks (e.g. K=8/16/32) from 48/class reference image patch tokens using a deterministic offline clustering/representative selection on surrogate features only. For every adversarial patch token, contrast its smooth nearest-target-patch-center similarity against nearest-source-patch-center similarity. With masks `M`, mean over valid patches. This gives `L_global_local = (1-lambda)*L_global + lambda*L_patch`. Start with small `lambda` and K, and include global-only and patch-only controls.

Beware shortcut matching to background patches. Evaluate coverage, per-image spatial masking, feature token stability and whether patch loss improves target-vs-source semantics rather than painting every region with generic textures. If a given proxy exposes no valid patch grid/tokens, explicitly mark incompatibility instead of silently using a pooled vector.

Recommended controlled primary ladder:
A0 L0 single/deepest; A1 L1 centroid; A2 L1+L2 distributional reference; A3 L1+multi-layer; A4 L1+patch; A5 combined best TWO independently validated components. Do not launch every Cartesian combination blindly.

## 4. Gradient and perturbation optimization ablations (run AFTER loss diagnosis)

**G0** equal mean of per-image L1-normalized proxy gradients, as V6.

**G1** proxy subset selection: contrast CLIP, SigLIP2, DINOv2, Gemma, InternVL, Qwen standalone on **the same images**; use held-out surrogate models to choose 2 or 3 complementary proxies. The real target cannot drive the choice. Record surrogate class-margin efficacy and cross-proxy gradient cosine.

**G2** adaptive weights via cross-surrogate one-step improvement. For trial proxy gradient `g_i`, form `candidate_i=project(x - alpha*sign(g_i))`; evaluate other surrogate **representation losses** `L_j(candidate_i)`, compare with `L_j(x)`, compute weights `softmax_i(mean_{j!=i}[L_j(x)-L_j(candidate_i)]/T)`. Apply `stop_gradient` to the weight calculation; do not backprop second-order through trial candidates. This is AdaEA-inspired only, *not* reproduction of AdaEA classification loss or its scores. Account for extra surrogate forwards.

**G3** PCGrad-inspired projection: if `g_i·g_j<0`, project away the component opposing original `g_j`; deterministically order pair projections, record how often direction is near-zero and whether representation loss actually decreases. Project **L1-normalized pixel image gradients**, not network parameter gradients. Audit whether PCGrad hurts the strongest single surrogate; no assumed benefit.

**G4** spatial consensus filter and sign-consistency. Compare equal fusion and G2/G3 first, then optional conservative masking based on per-pixel sign agreement. Beware overly suppressing useful gradients.

**X1** DI²-style stochastic differentiable resize-and-pad, optionally multi-crop/multi-scale EOT; reverse gradients back to the *same original* 224px attack tensor, and project final original image to epsilon=16/255. Same optimizer steps and total forwards must be reported.

**X2** TI convolutional smoothing or gradient variance tuning: measure actual extra forwards/backwards per step and compare same-compute baselines. EOT seeds/restarts are fixed on the surrogate dev set, not selected after target evaluation.

**X3** stepped budget 50/100/200, and multiple surrogate-selected initial perturbations, *separate benchmark tables* from matched 50-step results.

## 5. Resource-safe integration architecture

Build `src/run_v7_attack.py` by factoring V6 `run_recovery_ensemble.py`: `--loss-mode`, `--reference-aggregation`, `--layers`, `--patch-K`, `--fusion-mode`, `--input-diversity`, `--seed`, `--steps`, `--output-dir`. Need a `src/generate_v7_prototypes.py` that stores per-model/revision/layer/seed/reference-ID hashes and token cluster banks. Do not rewrite the known-correct V6 fixed-cohort loader/PNG evaluation needlessly.

For multi-layer outputs, avoid unnecessarily repeated model loads: register forward hooks to return multiple layer activations from one vision pass if possible, but verify exact depth/tap. For multi-proxy under A100 80GiB, compute each proxy's gradients sequentially with BF16 models and CPU/GPU offload, accumulate FP32 pixel gradients, then fuse. All 15 model checkpoint BF16 audits already passed V6; re-audit every new load and any additional model. Preserve 48 class references and 30 attacked images per pair.

Launcher `run_all_v7.sh`: CLI --preflight, --smoke, --pilot, --full, --resume; per-method/target/pair/seed/budget state and deterministic image hashes. Start with CPU unit tests, then 2-image gradient/PNG smoke for A0, L1, L2, L3, L4, G2, G3, X1. Fail preflight on NaN, zero gradient, wrong tap, quantization, mixed precision or epsilon failure.

## 6. Evaluation design and no target-feedback search

Use **separate surrogate-held-out models** (e.g. train on one subset of open-weight surrogates and validate transfer to disjoint open-weight models) and a dev image set independent of the 150 locked target-test images to select:
- feature layer(s), temperature/margin/K and loss weights;
- proxy count and identities;
- adaptive weighting hyperparameters, conflict variants, DI probability and seed.

Then freeze selected algorithm/weights/perturbation generation and generate new target-test adversarial images using the locked V6 cohort. Evaluate each frozen batch with the unchanged target prompt/parser, ideally once at the end. Do not consult target-model outcomes to choose subsequent algorithms.

Report for each target in Qwen27B, InternVL14B, Gemma31B separately and each T01,T03,T05,T07,T09:
- exact target successes /30, ASR, clean-correct ASR, output-class histograms and off-target class change;
- minimum cell ASR, `all_15_cells_above_50pct`, target-wise pooled ASR;
- single-proxy baselines tested with **the same final optimized loss/compute budget**, no cherry-picked weak comparison;
- surrogate class margins and per-layer / per-patch change (including hard failed classes);
- BF16, epsilon, fixed cohort checks, attack forward/backward counts/runtime, multi-seed intervals, all failed ablations.

Acceptance goal is >=15/30 in ALL 15 cells. This is difficult; at the moment no new V7 scientific results exist. A failure is a real result: do not change class pairs, select only easy samples, inflate denominator, use target predictions to guide changes or present untargeted transfer as targeted success.

## 7. Sources and distinction from prior art

1. **Everywhere Attack** (AAAI 2025) introduces localized plus global *targeted* transfer; our proxy patch-center matching is **not** claimed as an exact implementation. https://ojs.aaai.org/index.php/AAAI/article/view/33061
2. **GLEAM** (ICCV 2025) uses global-local transformations in VLP attacks; our source/target image-prototype losses and setting differ. https://openaccess.thecvf.com/content/ICCV2025/html/Liu_GLEAM_Enhanced_Transferable_Adversarial_Attacks_for_Vision-Language_Pre-training_Models_via_ICCV_2025_paper.html
3. **MPCAttack** (CVPR 2026) uses BOTH visual image and language text semantic features for collaborative attack; our core is image-only and must not be represented as that algorithm. https://openaccess.thecvf.com/content/CVPR2026/html/Li_Multi-Paradigm_Collaborative_Adversarial_Attack_Against_Multi-Modal_Large_Language_Models_CVPR_2026_paper.html
4. **AdaEA** (ICCV 2023) weights classifier-ensemble gradients based on cross-model lookahead; ours must substitute visual representation loss. https://openaccess.thecvf.com/content/ICCV2023/html/Chen_An_Adaptive_Model_Ensemble_Adversarial_Attack_for_Boosting_Adversarial_Transferability_ICCV_2023_paper.html
5. **Transform to Transfer** (CVPR 2026) explores adaptive transform diversity/integrated gradients, but evaluates VLP, not our targeted 10-way 3-MLLM fixed cohort. https://openaccess.thecvf.com/content/CVPR2026/html/Li_Transform_to_Transfer_Boosting_Adversarial_Attack_Transferability_on_Vision-Language_Pre-training_CVPR_2026_paper.html
6. **IPGA** (arXiv 2025) studies intermediate projector guidance with Q-Former; our listed MLLMs should not be falsely assumed to contain Q-Former. https://arxiv.org/abs/2508.13739

If a method closely follows prior work, cite properly; the paper contribution should arise from experimentally justified new mechanisms, not simply stacking tricks.
