# Initial mechanism findings

Exploratory only:10 previously inspected development images per condition, evaluated by3 held-out checkpoints. These are30 correlated model-image observations, not30 independent images. No final-target or commercial output selected these arms.

| Composition | Intervention | Status | Hits /30 observations |
|---|---|---|---|
| A | reference | COMPLETE | 29/30 |
| A | raw_gradient_mean | COMPLETE | 28/30 |
| A | single_75 | COMPLETE | 26/30 |
| A | middle_pair | COMPLETE | 29/30 |
| A | late_triple | COMPLETE | 28/30 |
| B | reference | COMPLETE | 28/30 |
| B | raw_gradient_mean | PARTIAL | N/A/30 |
| B | single_75 | PARTIAL | N/A/30 |
| B | middle_pair | PARTIAL | N/A/30 |
| B | late_triple | QUEUED | N/A/30 |

## Existing DI removal results

Composition A: DI-on 29/30; off 16/30. Paired removal delta -43.3%; exploratory image-cluster95% interval [-63.3%,-26.7%].
Composition B: DI-on 28/30; off 21/30. Paired removal delta -23.3%; exploratory image-cluster95% interval [-40.0%,-6.7%].

## Balance and proxy evidence already available

Reuse the completed13-composition ratio study (13 matched ratio comparisons available), and the original five-ratio bounded diagnostic. Do not rerun either automatically. Balanced coefficients are interpretable but do not imply balanced pixel gradients.
For proxy complementarity use matched member singles from that same composition study. Low CKA is a candidate predictor, not evidence of a causal benefit or pixel-gradient orthogonality. No new CKA/RSA or fixed-view target representation extraction was performed in this CPU pass.

## What should determine the next larger experiment?

* DI: preserve the completed signal; no further DI search now.
* Gradient normalization: examine the paired score together with raw norm dominance/direction measurements. A small score difference does not prove equivalence or a necessary mechanism.
* Layers: determine whether75% alone,50/75% or50/75/100% retains the four-layer effect. These arms do not identify a globally best subset; follow up with a small equal-count identity comparison if warranted.
* Confirm promising contrasts on an independently frozen cohort and predeclared seeds before expanding original full tables. Do not choose a configuration using final-target feedback.
* Compare the same images and report direction-specific outcomes and image-cluster uncertainty; any later many-contrast inferential analysis needs multiplicity handling.

## Validation scope

CPU checks cover 38 attack batches, 76 clean/adv image pairs, 18 native validation files and 36 strict-BF16 load records available at this snapshot. Bytewise pixel bounds, PNG hashes, cohort uniqueness,50-step settings, no target access and native hit numerators were checked. Missing rows remain missing.
Optimization CSV endpoints use different DI views and different layer-specific objectives; do not interpret their raw loss differences as comparable fixed-view progress.
Raw L1 share is a magnitude diagnostic, not signed causal attribution. Source condition identities and per-image observations remain in the companion reports.
