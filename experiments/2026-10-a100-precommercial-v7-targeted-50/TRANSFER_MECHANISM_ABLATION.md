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

## Primary controlled comparisons

| ID | Change from reference | Question | Important control |
|---|---|---|---|
| M0 | None | Reference transfer level | Same images/seeds/reference bank |
| M1 | Deepest layer only | Does aggregation across depths matter? | Keep DI and same proxies |
| M2 | DI off | Does input transformation prevent surrogate overfitting? | Keep all four layers |
| M3 | Deepest only AND DI off | Are depth aggregation and DI complementary? | Complete2x2 with M0/M1/M2 |
| M4 | Pull only, weights1/0 | Is explicit source repulsion needed? | Same four layers and DI |
| M5 | Push only, weights0/1 | Does source removal suffice for targeted movement? | Track off-target changes too |
| M6 | Each ensemble member alone | Does combining proxies add transferable directions? | All members, not weakest single |
| M7 | Leave one proxy out | Which member contributes and which is redundant? | Same remaining model order |
| M8 | Remove per-proxy L1 normalization | Does gradient scale dominance matter? | Record effective contribution sizes |
| M9 | Momentum0 | Does accumulating directions matter? | Needs isolated runner; original runner hardcodes1 |
| C0 | Clean images | Natural target-label hit rate | Not assumed zero |
| C1 | Fixed random perturbation within same bound | Does semantic optimization exceed random perturbation? | Fixed seed, no success selection |

M8 is a scale-dominance intervention, not a pure proxy-identity comparison.
Momentum/sign updates can confound gradient scale and accumulation, so inspect
update directions and step trajectories. Do not silently call the unchanged
momentum1 runner a momentum0 implementation.

Stage1 prioritizes M0-M7 and C0/C1. M8/M9 follow only if gradient diagnostics
justify them. No TI or patch component is added to the main method in this study.
Restore an ablated component using the same seed/image/proxy order; compare the
restored result to the locked reference to test reproducibility. This supports
dependence under this protocol, not a universal necessity claim.

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
