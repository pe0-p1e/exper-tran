# Representation-attack baselines

Prepared on 2026-10-20. Existing original13 full test work keeps GPU priority.
Preparation does not launch a new full matrix or a commercial request.

## Comparisons already in progress

The original13 runner exports the new four-layer Pull/Push .5/.5+DI, historical
A3 margin/patch+DI and historical V6 deepest Pull/Push rows. Existing measurements
are reused with cohort/settings/provenance checks. These are method-level
comparisons: several factors differ. They do not isolate a coefficient effect.
Matched member singles are required before claiming an ensemble gain. E01/E04/E07
alone do not supply every member of every ensemble.

## Controlled optimizer baselines prepared

All use pure cosine Pull/Push .5/.5, zero CLS/patch, equal independently
L1-normalized proxy gradients, BF16 weights, eps16/255, 50 steps, alpha1/255,
momentum1, seed42, one random start, 224px canvas,48 reference images/class.

| Arm | Visual depths | DI | TI | Isolated comparison |
|---|---|---|---|---|
| R_MI_deep | 100% | off | off | Reference depth control |
| R_MI_four | 25/50/75/100% | off | off | Depth aggregation |
| R_DI_MI_four | four | on | off | Input diversity |
| R_TI_MI_four | four | off | on | Gradient smoothing |
| R_DI_TI_MI_four | four | on | on | DI/TI interaction |
| Matched singles | four | on | off | Each ensemble member alone |

Three fixed CLIP + two cross-family VLM pools give24 target-specific display
conditions (15 optimizer/depth arms +9 matched singles). This is a preparation
matrix, not24 completed results and not a new authorized full-test queue.
The ten development images are the previously inspected dev positions2:4 per
direction. They are not a fresh holdout. A later confirmatory cohort is needed.

These are **representation-objective adaptations** of momentum, DI and TI;
they are not faithful classifier-logit reproductions of the cited papers.
Our TI kernel is5x5 Gaussian sigma1 after equal fusion and before momentum.
The unchanged runner always uses momentum1, so it is not labeled plain PGD.
Report exact DI implementation, model forwards/backwards, preprocessing, layer
IDs and runtime; the same number of steps is not identical compute for one
versus three proxies. Add a separately labeled compute-budget comparison before
making efficiency claims.

References: [Momentum, CVPR2018](https://arxiv.org/abs/1710.06081),
[Input diversity, CVPR2019](https://arxiv.org/abs/1803.06978),
[Translation invariance, CVPR2019](https://arxiv.org/abs/1904.02884).

## External method: FOA-Attack

[Official source](https://github.com/jiaxiaojunQAQ/FOA-Attack) reviewed separately.
Official global/local feature alignment includes optimal transport. Our earlier
patch-softmax term is not that algorithm. Status: SOURCE_REVIEW_ONLY; no faithful
FOA implementation or measured result claimed. Before execution, preserve a
pinned source revision, audit preprocessing/target-image semantics, clustering,
OT solver/marginals/regularization, dynamic weighting and optimizer budget.
Changing from its reference-image setting to48 image-class references must be
documented as an adaptation. Never run upstream commercial/key-loading entry
points. Any upstream weight precision must be replaced by audited BF16 locally.

Pinned revision: `6aa9ad463a1e812b7f77ff2e5c4b1712e7c78c4b` (file hashes in
`baselines/foa_source_lock.json`). Its `FOAttack.py` entry point retries cluster3
then5 conditional on target-side success. That feedback loop is incompatible
with our protocol: fix any clustering choice on a separate surrogate development
set before target evaluation, and disclose the deviation. No upstream script
was executed, and source review does not establish reproduction correctness.

## Entry point and execution safety

`bash experiments/2026-10-a100-precommercial-v7-targeted-50/prepare_baselines.sh`
only writes the immutable protocol, configs, matrix and hash-validated existing
original13 inventory under persistent `outputs/a100_precommercial_v7_targeted50/representation_baselines/`.
It loads no GPU models. `--condition ID --smoke` is an opt-in2-image/2-step attack;
`--condition ID` is an opt-in10-image development attack. Both refuse GPU ownership
while the current scheduler lock is held. There is no final target evaluator in
this preparer. Smoke outputs have separate identities and never count as final.

Before scaling, run real GPU smoke, validate epsilon/PNG hashes, gradient flow,
BF16 audit and resume; then evaluate frozen PNGs with the existing held-out
surrogate evaluator under the shared scheduler. Target-family exclusion of attack
proxies does not establish exclusion of that family from historical validators.
No final-target or commercial outcomes select algorithms or hyperparameters.
