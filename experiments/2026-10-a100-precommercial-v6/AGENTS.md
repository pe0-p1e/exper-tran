# AGENTS.md — A100 Pre-Commercial V6

This directory defines the authoritative instructions for completing all open-source experiments before any commercial-model evaluation.

## Scope

Complete every experiment in this plan on the branch `experiment/a100-precommercial-v6`.

DO:
- finish Single Proxy;
- finish Embedding Layer experiments;
- finish Multiple Proxy exactly as specified below;
- finish Pull/Push ablation;
- finish representation/mechanism analysis, including joint PCA/t-SNE, variance, asymmetry, CKA/RSA, representation shift, and correlations;
- make the code resumable, reproducible, and suitable for an NVIDIA A100 80GB.

DO NOT:
- call or evaluate any commercial API/model;
- implement OpenAI/Anthropic/Google commercial evaluation;
- redesign the Multiple Proxy experiment into Phase A/B/C;
- change the scientific objective, class pairs, image count, loss weights, or TASR definition without explicit user instruction;
- use target-model gradients, target logits, target embeddings, or target outputs during adversarial-example generation;
- overwrite V5 outputs.

The commercial-model sheet is out of scope.

---

## Scientific protocol

### Attack threat model

The attack is surrogate-only.

For every attack:
1. Load proxy model(s).
2. Generate adversarial images using only proxy-side information.
3. Save/freeze the adversarial PNGs.
4. Unload proxies if needed.
5. Load target model.
6. Evaluate the frozen adversarial PNGs.
7. Target-side representations may be extracted only for post-hoc analysis.

No target information may influence attack optimization.

### Final attack objective

Use normalized Pull/Push weights that sum to one:

```
w_pull = 0.75
w_push = 0.25
```

For proxy representation `z_adv`, target-class prototype `mu_t`, and source-class prototype `mu_s`, all in the proxy representation space:

```
L = 0.75 * (1 - cosine(z_adv, mu_t))
  + 0.25 * (1 + cosine(z_adv, mu_s))
```

There is NO classification-loss term.

The historical `1.5/0.5` form is the same ratio but must not appear in new configs/results. New V6 configs, metadata, tables, and reports must use `0.75/0.25`.

### Attack hyperparameters

Use, unless an existing V5 implementation requires an equivalent representation:

- L_inf epsilon: `16/255`
- steps: `50`
- step size: `1/255`
- momentum: `1.0`
- random start: enabled
- seed: `42`
- attack canvas: `224 x 224`
- proxy precision on A100: BF16 where supported
- representation pooling: reuse the existing V5 spatial-token mean-pooling semantics
- main-experiment proxy tap: deepest visual block
- target-side post-hoc representation tap: deepest visual block

If a visual backbone has `N` blocks, deepest block index is `N-1`. Resolve this from the loaded model/config when possible; record the resolved index in metadata rather than silently assuming it.

### TASR

The only main transfer metric called TASR is:

```
TASR =
  (# samples where proxy targeted attack succeeds AND target predicts the same target class)
  /
  (# samples where proxy targeted attack succeeds)
```

For a single proxy, proxy success means that proxy predicts the specified target class.

For a multi-proxy ensemble, store per-proxy hit masks. Use **all-proxy targeted success** as the primary conditioning event for ensemble TASR:
```
ensemble_proxy_success = all participating proxies predict the specified target class
```
Also export any-proxy and majority-proxy diagnostics, but do not call them the main TASR.

Never relabel the clean-valid denominator rate as TASR. It may be exported as a diagnostic only.

---

## Dataset

Use the ImageNet-1K diverse-10 setup.

Classes:
1. goldfish
2. monarch butterfly
3. pineapple
4. acoustic guitar
5. laptop
6. espresso
7. volcano
8. rocking chair
9. soccer ball
10. school bus

Main source -> target pairs:
1. goldfish -> monarch butterfly
2. pineapple -> acoustic guitar
3. laptop -> espresso
4. volcano -> rocking chair
5. soccer ball -> school bus

Use exactly **30 attack images per directed source-target pair** for the planned experiments.

Reference bank:
- 48 reference images per class;
- reference images must be disjoint from attack candidates.

Candidate bank:
- keep at least 64 candidate images per class;
- freeze deterministic manifests;
- never silently replace images between compared settings.

For model-comparison experiments, use the same image IDs whenever the required clean-valid intersection contains >=30 images. If the intersection contains fewer than 30, stop that cell and report the shortfall instead of silently changing the cohort.

For asymmetry/mechanism analysis, additionally run the five reverse directions with the same protocol:
- monarch butterfly -> goldfish
- acoustic guitar -> pineapple
- espresso -> laptop
- rocking chair -> volcano
- school bus -> soccer ball

These reverse runs are for mechanism/asymmetry analysis and do not change the five forward-pair main tables.

---

## A100 80GB execution

Target hardware: NVIDIA A100 80GB.

Retain the V5 environment unless a model requires a minimally necessary compatibility fix:
- Python 3.11
- PyTorch 2.5.1 + CUDA 12.4 class environment
- BF16 proxies where supported

Use V5 memory calibration as the base:
- candidate batches: 8, 16, 24, 32, 48, 50;
- run a real gradient step;
- select the largest batch with peak reserved VRAM <= 75 GiB;
- leave safety headroom;
- record allocated/reserved peak VRAM and runtime.

Do not load the target simultaneously merely to fill VRAM. Scientific isolation of proxy attack and target evaluation is more important than nominal utilization.

If `lambda_cls == 0`, the attack loop must skip language/classification forward passes exactly as V5 already supports.

All jobs must be resumable by pair / class transition / layer / ablation arm / ensemble configuration.

### Required performance optimization

Experiment completion time matters. Optimize the implementation before launching the full matrix, while preserving exact numerical/scientific semantics.

Required optimizations:

1. **Skip unused language forwards**
   - When classification-loss weight is zero, do not call the language head/decoder during iterative attack steps.
   - Only run proxy classification when needed for clean screening and final proxy-success evaluation.

2. **Use BF16 and inference/autocast correctly**
   - Use BF16 for supported model weights/forwards on A100.
   - Keep gradient-bearing image tensors in a numerically safe dtype as required.
   - Use `torch.inference_mode()` / `no_grad()` for all non-attack evaluation, reference extraction, clean screening, target evaluation, CKA/RSA feature extraction, and plotting data generation.

3. **Cache all invariant reference embeddings**
   - Compute each model/class/reference-bank embedding once per model revision + layer + preprocessing configuration.
   - Cache source/target prototypes and reusable clean embeddings on disk.
   - Never recompute the 48-image reference bank inside each transition or each attack step.

4. **Cache clean screening and manifests**
   - Run clean prediction/screening once per model + dataset manifest.
   - Reuse the resulting clean-valid masks across experiments that share the same model and cohort definition.

5. **Reuse loaded models aggressively**
   - Order jobs to minimize checkpoint reloads.
   - For Single Proxy and layer sweeps, group all transitions/layers that share the same proxy before unloading it.
   - For target evaluation, batch all frozen adversarial examples for the same target and evaluate them in as few model loads as possible.
   - Do not repeatedly load the same model for each class pair if memory can be safely reused.

6. **Batch reference/analysis extraction**
   - Extract target/proxy embeddings for DeltaR, CKA, RSA, variance, PCA/t-SNE inputs in batches.
   - Save high-dimensional embeddings once and run all post-hoc analyses from cached embeddings.
   - PCA/t-SNE, variance, asymmetry, and correlation code must not rerun model inference if cached embeddings already exist.

7. **Avoid redundant PNG generation and decoding**
   - Save each adversarial image once under a deterministic artifact key.
   - Reuse frozen adversarial images for target evaluation and all post-hoc analyses.
   - Prefer batched tensor pipelines during attack; PNG encoding is an artifact/export step, not part of the inner optimization loop.

8. **A100 batch-size calibration**
   - Calibrate the largest safe batch per proxy/model family using a real gradient step.
   - Target <=75 GiB peak reserved VRAM.
   - Persist calibrated batch sizes so later jobs do not recalibrate unnecessarily.

9. **Model-aware scheduling**
   - Group jobs by currently loaded proxy/target to reduce load/unload overhead.
   - When multiple independent lightweight post-hoc tasks fit safely, they may be parallelized.
   - Do not run concurrent GPU jobs that cause memory pressure, nondeterministic OOMs, or change attack semantics.

10. **DataLoader / CPU pipeline**
    - Use pinned memory and a sensible number of workers for image decode/transform.
    - Avoid repeated preprocessing of the same reference images when tensors/features can be cached.
    - Ensure CPU I/O does not starve the A100.

11. **Optional PyTorch optimizations, only after correctness check**
    - TF32 may be enabled for non-sensitive matmul paths if outputs/attack decisions remain unchanged within tolerance.
    - `torch.compile` may be used only for stable repeated modules after a correctness and speed benchmark; do not spend substantial time compiling highly dynamic generation code.
    - Do not enable an optimization merely because it exists; retain it only if measured wall-clock time improves.

12. **No scientific shortcuts**
    - Performance optimization must not reduce steps, images, reference count, precision below the specified protocol, model size, layer count, or number of experiment cells.
    - Do not replace models with smaller variants for speed.
    - Do not reuse attack outputs across scientifically different loss/layer/ensemble settings.

### Performance benchmark and acceptance

Before the full matrix, benchmark at least one representative 30-image attack cell before and after optimization.

Record:
- wall-clock seconds per attack step/cell;
- images/second where meaningful;
- model load time;
- reference-embedding extraction time;
- target-evaluation throughput;
- peak allocated/reserved VRAM.

Create:
`outputs/a100_precommercial_v6/audits/performance_benchmark.json`

and summarize the optimizations and measured speedups in the final report.

Prefer optimizations that reduce repeated model forward passes and model reloads; these are expected to dominate micro-optimizations.

---

## Model naming and main deepest layers

Use exact experiment-facing names:

- Qwen3.5-2B
- Qwen3.5-4B
- Qwen3.5-9B
- Qwen3.5-27B
- InternVL3.5-2B-HF
- InternVL3.5-4B-HF
- InternVL3.5-8B-HF
- InternVL3.5-14B-HF
- Gemma 4 E2B-it
- Gemma 4 E4B-it
- Gemma 4 26B-A4B-it
- Gemma 4 31B-it
- CLIP ViT-L/14
- SigLIP2-So400m-patch14-384
- DINOv2-Large

For every newly added model:
1. register the exact backend/Hugging Face identifier;
2. pin and record an immutable revision/commit;
3. audit actual vision depth;
4. verify image preprocessing;
5. verify representation extraction;
6. verify one-step gradients for proxies;
7. verify clean generation/classification parsing for targets.

Known main taps:
- Qwen3.5-2B / 4B: deepest block 23;
- InternVL3.5-2B-HF / 4B-HF: deepest block 23;
- Gemma 4 E2B-it / E4B-it: deepest block 15.

For larger/new models, resolve deepest block from the loaded config and record it. Do not rely on an unverified spreadsheet number.

---

## Single Proxy — exact matrix

Run all five forward class pairs x 30 images for every row.

Cross-family:
1. Qwen3.5-2B -> InternVL3.5-4B-HF; proxy deepest layer 23
2. Qwen3.5-2B -> Gemma 4 E4B-it; proxy deepest layer 23
3. InternVL3.5-2B-HF -> Qwen3.5-4B; proxy deepest layer 23
4. InternVL3.5-2B-HF -> Gemma 4 E4B-it; proxy deepest layer 23
5. Gemma 4 E2B-it -> Qwen3.5-4B; proxy deepest layer 15
6. Gemma 4 E2B-it -> InternVL3.5-4B-HF; proxy deepest layer 15

Intra-family:
7. Gemma 4 E2B-it -> Gemma 4 E4B-it; proxy deepest layer 15
8. Gemma 4 E2B-it -> Gemma 4 26B-A4B-it; proxy deepest layer 15
9. Qwen3.5-2B -> Qwen3.5-4B; proxy deepest layer 23
10. Qwen3.5-2B -> Qwen3.5-9B; proxy deepest layer 23
11. InternVL3.5-2B-HF -> InternVL3.5-4B-HF; proxy deepest layer 23
12. InternVL3.5-2B-HF -> InternVL3.5-8B-HF; proxy deepest layer 23

Per cell export:
- N clean-valid;
- N proxy targeted successes;
- N transfer successes among proxy successes;
- TASR;
- mean/median target representation shift Delta R;
- CKA;
- RSA;
- runtime;
- peak VRAM;
- exact proxy/target revisions;
- resolved proxy/target representation taps.

Do not replace the 30-image main design with V5 full-90.

---

## Embedding Layer experiment — exact matrix

The independent variable is proxy embedding depth. One row/result per layer.

Use five normalized depths:
- 1%
- 25%
- 50%
- 75%
- 100%

For a 24-block proxy, use indices:
- 0, 5, 11, 17, 23

For a 16-block proxy, use:
- 0, 3, 7, 11, 15

If actual depth differs, compute equivalent normalized indices deterministically and record them.

Run these six proxy-target pairs:
1. Qwen3.5-2B -> Gemma 4 E4B-it
2. InternVL3.5-2B-HF -> Qwen3.5-4B
3. Gemma 4 E2B-it -> InternVL3.5-8B-HF
4. Qwen3.5-2B -> Qwen3.5-4B
5. InternVL3.5-2B-HF -> InternVL3.5-4B-HF
6. Gemma 4 E2B-it -> Gemma 4 E4B-it

For every layer, run the same five forward class pairs x 30 images.

Only the proxy layer changes. Keep:
- data;
- target;
- target analysis layer;
- loss;
- optimizer;
- seed;
- image IDs;
- prompts;
- all other hyperparameters
fixed.

Export one long-form CSV row per:
`proxy-target x class transition x layer depth`.

Primary layer-study outputs:
- TASR;
- Delta R;
- CKA;
- RSA;
- proxy-success denominator;
- actual block index.

Also produce:
- TASR vs normalized depth plots;
- CKA vs normalized depth;
- TASR vs CKA scatter.

---

## Multiple Proxy — DO NOT REDESIGN

Use the exact two blocks below. Do not convert them into Phase A/B/C.

### Block 1: same-family proxy scaling / number

Target: Qwen3.5-27B
1. InternVL3.5-2B-HF
2. InternVL3.5-2B-HF + InternVL3.5-4B-HF
3. InternVL3.5-2B-HF + InternVL3.5-4B-HF + InternVL3.5-8B-HF

Target: Gemma 4 31B-it
4. Qwen3.5-2B
5. Qwen3.5-2B + Qwen3.5-4B
6. Qwen3.5-2B + Qwen3.5-4B + Qwen3.5-9B

Target: InternVL3.5-14B-HF
7. Gemma 4 E2B-it
8. Gemma 4 E2B-it + Gemma 4 E4B-it
9. Gemma 4 E2B-it + Gemma 4 E4B-it + Gemma 4 26B-A4B-it

### Block 2: ensemble-combination study

Target: Qwen3.5-27B
10. InternVL3.5-2B-HF
11. InternVL3.5-2B-HF + Gemma 4 E2B-it
12. InternVL3.5-2B-HF + Gemma 4 E2B-it + CLIP ViT-L/14
13. InternVL3.5-2B-HF + Gemma 4 E2B-it + CLIP ViT-L/14 + SigLIP2-So400m-patch14-384
14. InternVL3.5-2B-HF + Gemma 4 E2B-it + CLIP ViT-L/14 + DINOv2-Large

Retain the spreadsheet comparison rows for:
- Qwen3.5-2B;
- Qwen3.5-2B + Qwen3.5-4B;
- Qwen3.5-2B + Qwen3.5-4B + Qwen3.5-9B
against Gemma 4 31B-it;

and:
- Gemma 4 E2B-it;
- Gemma 4 E2B-it + Gemma 4 E4B-it;
- Gemma 4 E2B-it + Gemma 4 E4B-it + Gemma 4 26B-A4B-it
against InternVL3.5-14B-HF.

Run all five forward class pairs x 30 images for every unique row.

### Multi-proxy gradient fusion

Do not concatenate embeddings from heterogeneous models.

For each proxy `i`:
1. compute its own Pull/Push loss in its own representation space;
2. compute per-image pixel gradient `g_i`;
3. L1-normalize `g_i` per sample;
4. average normalized gradients with equal proxy weights;
5. feed the combined gradient into the shared momentum/sign update.

Save per-proxy gradient norms and pairwise gradient cosine similarities.

Also compute pairwise proxy CKA/RSA on the same clean reference/calibration image set where meaningful. These are analysis metrics, not direct alignment losses.

---

## Ablation Study

Run only Method Design components / weight allocation. No CLS loss.

Unique Pull/Push ratios:
1. 1.00 / 0.00
2. 0.25 / 0.75
3. 0.50 / 0.50
4. 0.75 / 0.25
5. 0.00 / 1.00

The spreadsheet's repeated final-method row `0.75/0.25` must reuse the same run; do not rerun it as a sixth arm.

Run ablation on:
- Gemma 4 E2B-it -> Gemma 4 E4B-it
- Qwen3.5-2B -> Qwen3.5-4B

Use each proxy's deepest visual block.

For every ratio and model pair:
- same five forward class pairs;
- 30 images each;
- same manifests;
- same seed and optimizer settings.

Export detailed pair-level and class-pair-level results, not only pooled totals.

For push-only, if proxy targeted-success denominator is zero, TASR is undefined/NA, not 0%.

---

## Representation/mechanism analysis

Complete this after attack/evaluation artifacts exist.

### Target representation shift

In the target encoder space:

```
delta_pull =
  cos(z_adv, target_center) - cos(z_clean, target_center)

delta_push =
  cos(z_clean, source_center) - cos(z_adv, source_center)

DeltaR = delta_pull + delta_push
```

Store per-image and aggregated mean/median DeltaR.

### Class variance / dispersion

Use L2-normalized target-space class-reference embeddings.

Primary dispersion:
```
variance_c = mean_i [1 - cosine(z_i, class_center)]
```

Also compute:
- covariance trace;
- effective rank.

Call the primary quantity class embedding dispersion in reports when possible.

### Clean target margin

```
clean_margin =
  cos(z_clean, target_center)
  - cos(z_clean, source_center)
```

Also compute adversarial margin and margin change.

If implementing normalized gap closure, define it separately and correctly. Do not duplicate margin_change under another name.

### Prototype distance

```
prototype_distance = 1 - cosine(source_center, target_center)
```

This is symmetric and therefore cannot by itself explain A->B vs B->A asymmetry.

### CKA and RSA

- CKA: compare proxy and target representations on the same clean images; different feature dimensions are allowed.
- RSA: compute within-model pairwise similarity/distance structures first, then correlate the flattened structures.

Do not directly subtract embeddings from different models.

### Joint PCA / t-SNE

For every unordered class pair used in asymmetry analysis, form ONE combined set in a single target encoder space:

```
source references
target references
source clean
source->target adversarial
target clean
target->source adversarial
```

L2-normalize first.

PCA:
- fit once on the combined set;
- transform all groups using the same fitted PCA;
- show clean -> adversarial movement arrows.

t-SNE:
- use the same combined set;
- optionally PCA-precompress to <=50 dimensions;
- fit t-SNE once;
- fixed random_state=42;
- use t-SNE only for qualitative visualization.

Never fit the two classes separately and overlay them.

### Asymmetry

For each unordered class pair, compare both directions:
- TASR A->B
- TASR B->A
- Delta TASR
- source/target dispersion
- clean margin
- DeltaR
- margin change
- prototype distance

Produce an `asymmetry.csv` and identify the largest directional gaps.

### Correlation matrix

Within each model pair, compute Spearman correlations over directed transitions for:
- TASR
- CKA
- RSA
- source variance/dispersion
- target variance/dispersion
- prototype distance
- DeltaR
- clean margin
- margin change
- normalized gap closure if properly implemented

Export CSV + heatmap.

Treat these as exploratory associations, not causal proof.

---

## Required output structure

Use a new output root and never overwrite V5:

```
outputs/a100_precommercial_v6/
  manifests/
  audits/
  batch_calibration/
  single_proxy/
    attacks/
    evaluations/
    summaries/
  layer_sweep/
    attacks/
    evaluations/
    summaries/
  multiple_proxy/
    attacks/
    evaluations/
    summaries/
  ablation/
    attacks/
    evaluations/
    summaries/
  analysis/
    embeddings/
    representation_shift/
    variance/
    asymmetry/
    pca/
    tsne/
    correlations/
  reports/
```

Every experiment cell must have:
- config snapshot;
- model revisions;
- data-manifest hash;
- seed;
- source/target labels;
- image IDs;
- actual layer index;
- proxy hit mask;
- target hit mask;
- TASR numerator/denominator;
- runtime;
- peak VRAM;
- completion status.

---

## Implementation order

1. Audit/extend model registry and immutable revisions.
2. Validate each newly added model with a 2-image smoke test.
3. Prepare/freeze reference and 30-image manifests.
4. A100 batch calibration.
5. Single Proxy complete matrix.
6. Embedding Layer complete matrix.
7. Multiple Proxy exact two-block matrix.
8. Ablation complete matrix.
9. Reverse-direction mechanism runs.
10. Joint PCA/t-SNE, variance, DeltaR, margin, asymmetry, CKA/RSA, correlation analysis.
11. Generate a final open-source-only report and machine-readable summaries.
12. Stop. Do not proceed to commercial models.

---

## Acceptance criteria

The task is not complete until:

- all 12 Single Proxy rows are complete for 5 x 30 images;
- all 6 layer-study pairs x 5 depths are complete for 5 x 30 images;
- every unique Multiple Proxy row above is complete for 5 x 30 images;
- both ablation model pairs x 5 unique ratios are complete for 5 x 30 images;
- reverse-direction mechanism data exist for the five unordered class pairs;
- joint PCA/t-SNE uses one combined fit per unordered pair;
- TASR uses the proxy-success conditional denominator;
- all perturbations satisfy L_inf <= 16/255;
- all main experiments use Pull/Push = 0.75/0.25 except ablation;
- no target information is used during attack generation;
- no commercial model/API is called;
- all model revisions, manifests, taps, runtime and VRAM are recorded;
- results can be resumed without redoing completed cells;
- summary CSVs and a final Markdown report are produced;
- tests/smoke checks pass and no result contains unexplained NaN/empty denominators.

When a cell cannot run because of model incompatibility, OOM, missing checkpoint, or insufficient clean-valid images, record the failure explicitly and continue independent cells. Never silently substitute another model, image cohort, layer, or metric definition.
