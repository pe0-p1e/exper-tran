# Prepared next-stage code

Latest order remains corrected recovery/report → proxy selection → new-category
generalization → commercial evaluation. Paper work is paused. These new modules
do not modify the source fingerprints of the currently running correction.

## Proxy selection

`run_proxy_selection.sh --prepare-only` refreshes the seven-subset table on CPU
while the main experiment runs. It marks missing/partial measurements instead
of inventing zeros. No GPU model is loaded by this preparation mode.

The scientific entry point, after the corrected campaign finishes, is
`run_proxy_selection.sh`. It refuses concurrent GPU ownership, checks the prior
campaign's completion/BF16 validation, freezes all21 conditions and attempts the
missing subset attacks/evaluations. Compatible Qwen/InternVL main/single results
and historical valid Gemma measurements are reused only after checking actual
PNG hashes, perturbations, image IDs, settings and raw-answer identities.

The generated tables contain all7 subsets per target, gains against the best
member single/best pool single, paired stratified bootstrap intervals, gradient
cosines, negative-cosine fraction, equal-mean cancellation, runtime/VRAM, and
clean-feature CKA/RSA. Clean features are reused by content/revision/layer key;
`measure_selection_clean.py` fills missing proxy-only features sequentially.
Degenerate statistics remain undefined rather than zero. Target score rankings
are descriptive and do not tune another attack on that same target.

## New categories and10/20/50 options

`category_data.py` accepts an ImageFolder dataset and a JSON catalog of exact
class IDs/names, excludes the original ten synsets, samples new classes and
images deterministically, copies canonical224px PNGs to persistent root storage,
and freezes nested dictionaries with48 references/class and30 images/direction.
Reference/attack overlap, duplicate canonical pixels, changed manifests and
unsafe file paths are rejected. Source/target pairs and shared references/images
are identical across dictionary sizes. The catalog/new images are not supplied
yet; the preparer reports an explicit dataset blocker until they are available.

Example catalog schema (not an actual selected class list):

```json
[{"id":"EXACT_DATASET_CLASS_ID","name":"class name"}]
```

Preparation command when data is available:

```bash
.venv-a100/bin/python experiments/2026-10-a100-precommercial-v7-targeted-50/src/category_data.py \
  --dataset-root /path/to/new-class-folders \
  --catalog /path/to/versioned_class_catalog.json \
  --class-counts 10 20 50
```

`run_category_generalization.sh --prepare-only` validates those actual manifests
and creates a proposed plan without GPU inference. Full execution is gated on
the completed proxy-selection measurements. It freezes the current V+A+B
composition from the framework, NOT the best actual-target row of the ablation
table. A future learned selector must instead supply an independently validated
rule; no target-feedback selection is silently introduced.

`CategoryVisionProxy` reuses the existing vision extraction/pooling and builds
class-wise caches shared by10/20/50. `category_attack.py` calls the unchanged
visual margin/local loss and equal normalized gradients, with the original
50-step/16-over255 budget. Separate real2-image/2-step smoke outputs precede each
full job and never count as scientific50-step results. Target evaluation occurs
after attacks freeze, uses strict BF16 and a C-class prompt/parser, and reports
all GT-selected samples. The original ten-class parser/runner is not changed.

These tests establish CPU contracts for class indexing beyond9, gradients at
12/50 classes, original-ten prompt equivalence, disjoint manifests, nested
dictionary identity, feature statistics and paired intervals. A real new-class
A100 smoke remains REQUIRED after the phase/data prerequisites are met. No
new-category or commercial success rates are claimed from this code preparation.

Each scientific runner writes its own completion audit and report, then archives
compact JSON/CSV/Markdown provenance and measurements to the current branch.
Canonical images, adversarial PNGs, feature arrays and reference caches remain
on persistent local storage. The category archive includes complete class/image
manifests so reported rows can be traced to the frozen cohort.
