# Removal and restoration study of targeted representation transfer

User priority,2026-10-20: remove components of the simple method and determine
when transfer weakens or disappears. This is a mechanism study, not a search for
a higher final-target score. This document is a plan; it does not start GPU jobs.
Historical result directories and scientific source locks remain unchanged.

## Frozen reference method

Four visual depths25/50/75/100%, uniform spatial-token mean Pull/Push .5/.5,
DI, independently per-image L1-normalized gradients followed by equal proxy
averaging, momentum1,50 steps, alpha1/255, eps16/255, seed42, one random start,
224px. All floating open-weight model weights BF16, no quantization. No language
loss or target feedback. Reference and GT-selected attack cohorts remain fixed
and disjoint. Evaluation follows freezing and hashing all PNGs.

## Ordered user override —2026-10-20

Investigate ONE mechanism at a time, in this order. The earlier proposal to start
with deepest-layer controls or a broad factorial grid is superseded. Keep the
frozen reference method as the common anchor. Later removals do not silently
accumulate: each comparison changes only its declared factor. If a leaner method
is later adopted, record a new version and retain bridges to the original anchor.

### 1. Input diversity

Compare reference DI-on versus DI-off with identical .5/.5 loss, four layers,
proxies, cohort, seed and budget. Reuse existing matching DI-on artifacts.
Previously completed .75/.25 DI diagnostics are background evidence, not an exact
replacement for the .5/.5 comparison. Track surrogate optimization versus target
movement to distinguish overfitting from simply poorer optimization. Do not tune
DI probability or add TI in this removal study.

### 2. Per-proxy gradient normalization

Keep the same reference DI setting and all other conditions. Compare equal mean
of individually L1-normalized pixel gradients versus equal mean of raw pixel
gradients; both retain momentum1 and sign projection. Record each proxy's raw
L1 norm, norm ratio, contribution dominance, fused direction, gradient cosine
and cancellation. A raw mean has equal numerical coefficients but not equal
influence. Use at least a multi-proxy condition; a single-proxy scalar change
alone is not a valid test of cross-proxy balancing.

### 3. Which layer, how many layers, and why?

First recover the previous single-layer sweep and its provenance. The comparator
is the previously identified best eligible single layer, NOT automatically the
deepest layer. Existing A3 single-layer and leave-one-layer-out results are
historical evidence; their loss differs from pure Pull/Push, so their scores
cannot be relabeled as current-method results. A single layer identified from
final-target scores is descriptive only: choose any deployable rule using
held-out surrogate development evidence, then freeze it before confirmation.

Separate layer identity from count. Inspect existing 1/25/50/75/100% taps and
actual block IDs. Compare the eligible dev-selected best single, a small
predeclared two-layer set, three layers and the existing four; at fixed count,
compare alternative layer identities rather than always taking a deeper prefix.
Use leave-one-layer-out measurements to test marginal contribution. Avoid an
exhaustive powerset. Every fixed proxy keeps its own actual layer mapping.

Explain selection using per-layer semantic class margins/dispersion, gradient
strength and alignment/conflict across layers, and cross-layer CKA/RSA on shared
images. Four evenly spaced taps are an engineering starting point, not a proven
optimum. Low redundancy is useful only if retained layers supply effective
semantic optimization. This study should allow the conclusion that fewer layers
suffice; no assumption that deeper or more layers must be better.

### 4. Pull/Push balance

Reuse the completed five-ratio .0/.25/.5/.75/1 grid and the completed13-composition
.5-versus.75 dev comparison after exact provenance checks. Only fill missing
matched controls; do not rerun the grid automatically. Interpret .5/.5 as equal
coefficients for attraction and repulsion, not proof of equal gradient influence
or a universal optimum. Report source and target gradient norms separately and
use a few matched contrasts to assess the role of each term.

### Proxy composition — a subsequent hypothesis study

Working hypothesis: individually effective proxies with less redundant
representation geometry may be more complementary. This is NOT a requirement
to maximize orthogonality. CKA/RSA compare relational geometry; raw embeddings
from different dimensional spaces have no directly meaningful shared dot-product
angle. Distinguish low representation similarity, pixel-gradient orthogonality,
and negative gradient conflict. Negative alignment may cancel useful updates.

Hold loss, selected layers, normalization, DI, model count and image IDs fixed;
compare more/less redundant compositions with similar standalone effectiveness
and account for compute. Use matched member singles and leave-one-proxy-out
results. Selection uses surrogate-only validation; actual target outcomes are
post-freeze explanatory observations. Investigate this after the ordered
component studies instead of launching another broad ensemble search now.

Clean and fixed bounded random-noise controls remain descriptive anchors.
Restore removed components with identical seed/image/model order when checking
reproducibility. No new momentum, TI, patch or commercial campaign is included
in this ordered scope. These interventions establish protocol-specific
dependence, not universal necessity or a guarantee that transfer disappears.

## Separate optimization failure from transfer failure

For every image, measure proxy loss decrease, proxy source/target cosine changes,
proxy target-vs-other margin, target post-hoc DeltaPull/DeltaPush/DeltaR and
target targeted hit. Proxy class predictions are diagnostics, never eligibility.

* Proxy objective improves poorly and target ASR drops: optimization failure may
  explain the drop; this alone does not isolate a transfer mechanism.
* Proxy objective improves comparably but target movement and ASR drop: evidence
  more specifically supports loss of cross-model transfer.
* Target representation moves but output does not: investigate downstream
  decision margins and class-direction geometry rather than declaring no transfer.

Save per-step pairwise proxy gradient cosine, negative-cosine fraction,
equal-mean cancellation, gradient magnitude and realized update direction.
Reuse CKA/RSA and target geometry where exact image/model/layer identities match.
PCA/t-SNE is a joint-fit illustration, never quantitative proof of alignment.
CKA/RSA correlation alone does not establish causality or guarantee transfer.

## Evidence reuse and selection boundaries

The completed small PP4 diagnostic has deepest/noDI, four/noDI, four/DI and
pull/push-ratio arms, but its layer/DI ladder uses .75/.25. Do not relabel those
as the new .5/.5 controls. The completed composition-ratio study supplies matched
.5/.5 singles and ensembles on the historically inspected dev10. Verify settings,
all source/model revisions, sample order, PNG hashes and scoring identity before
reuse. Historical A3 versus pure PP is a multi-factor method comparison.

Predeclare a small set of fixed compositions from the existing three-proxy
cross-family pools; do not choose the final-target highest-scoring row. Use the
same10 development images for a labeled exploratory stage, then a separately
frozen untouched cohort for confirmatory tests and seeds42/43/44 if feasible.
Target-family exclusion during attack does not imply family exclusion from all
historical development validators. Disclose both distinctions. Final-target
results describe the prespecified interventions; they do not select the next
method or commercial settings on the same test set.

## Reporting and interpretation

Report targeted hits/N, clean-correct conditional rate separately, source retained
and off-target rates, per-direction DeltaR, optimization progress, compute and
paired imagewise differences. Cluster resampling by image when the same image
is scored by multiple validators; do not count correlated observations as
independent images. Report intervals and multiplicity for multiple contrasts.

Do not define disappearance as a nonsignificant difference or an arbitrary ASR
threshold. Report (1) the paired loss relative to the full method and (2) residual
advantage relative to clean/random controls. To claim equivalence to a control,
predeclare a scientifically justified equivalence margin and acquire adequate
sample size; dev10 is exploratory and may be insufficient. A small aggregate
difference neither proves equivalence nor rules out direction-specific effects.

Desired outcome: identify reproducible conditions where surrogate optimization
still succeeds but cross-model transfer weakens, and conditions where restoring
a component restores transfer. If transfer persists across removals, report that
the component is unnecessary under these settings rather than forcing collapse.
