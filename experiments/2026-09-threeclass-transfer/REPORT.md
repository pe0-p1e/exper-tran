# Three-Class Directed Transfer: Loss Ablation, Tuning, and 50-Image Confirmation

## Executive summary

This experiment tested directed transfer among ImageNet goldfish (class 1),
monarch butterfly (class 2), and volcano (class 7), using three fixed
small-to-large model pairs: P14 (Qwen 2B→4B), P16 (InternVL 2B→4B), and P19
(Gemma E2B→E4B). All six directed transitions were evaluated.

Main findings:

1. In the initial 8-image ablation, adding CLS did not change aggregate TASR:
   both arms achieved 93/144 = 64.6%.
2. CLS was model-dependent: it helped P14 and P19, but removing it helped P16.
3. P16 tuning improved 8-image TASR from 41.7% to 64.6%.
4. The final 50-image confirmation achieved conditional TASR of 79.3% for
   P14, 54.0% for P16, and 85.7% for P19.

TASR is conditional on strict proxy targeted success; ASR is reported on all
500 clean-valid images per pair.

## Experimental design

The six directions were:

| ID | Transition |
|---|---|
| G01 | goldfish → butterfly |
| G02 | goldfish → volcano |
| G03 | butterfly → goldfish |
| G04 | butterfly → volcano |
| G05 | volcano → goldfish |
| G06 | volcano → butterfly |

This supports both fixed-source and fixed-target comparisons. Models were
frozen. The attack used an L-infinity budget of 16/255, step size 1/255,
random start, momentum 1, and 50 steps. Each class used a deterministic,
disjoint 48-image source/target reference bank. The 50-image run used a common
clean cohort per transition.

The pull-push loss was:

```text
L_rep = target_weight * (1 - cosine(e_adv, mu_target))
        + source_weight * (1 + cosine(e_adv, mu_source))
```

The two main arms were `CLS + pull/push` and `pull/push only`.

## 8-image loss ablation

The denominator is 48 images per pair (six directions × eight images).

| Pair | CLS + pull/push TASR | Pull/push-only TASR | CLS + pull/push ASR | Pull/push-only ASR |
|---|---:|---:|---:|---:|
| P14 | 36/48 = 75.0% | 34/48 = 70.8% | 36/48 = 75.0% | 35/48 = 72.9% |
| P16 | 20/48 = 41.7% | 25/48 = 52.1% | 23/48 = 47.9% | 27/48 = 56.2% |
| P19 | 37/48 = 77.1% | 34/48 = 70.8% | 40/48 = 83.3% | 38/48 = 79.2% |
| **All pairs** | **93/144 = 64.6%** | **93/144 = 64.6%** | **99/144 = 68.8%** | **100/144 = 69.4%** |

CLS is therefore not a uniform transfer booster.

## P16 tuning

The best P16 configuration was:

```text
lambda_cls = 0
lambda_rep = 0.5
target pull weight = 1.5
source push weight = 0.5
```

It achieved 31/48 = 64.6% TASR, versus 20/48 = 41.7% for the initial
full-loss arm.

### P14/P19 tuning-arm comparison

The automatic 8-image search also evaluated four pull/push settings for P14
and P19 across all six directions. Each row below aggregates 48 images.

| Pair | Arm | TASR | Proxy success | ASR |
|---|---|---:|---:|---:|
| P14 | full target-pull | 37/48 = 77.1% | 48/48 | 37/48 |
| P14 | full push-strong | 34/48 = 70.8% | 48/48 | 36/48 |
| P14 | second target-pull | 39/48 = 81.3% | 48/48 | 39/48 |
| P14 | second target-pull, half coefficient | 39/48 = 81.3% | 48/48 | 39/48 |
| P19 | full target-pull | 40/48 = 83.3% | 48/48 | 40/48 |
| P19 | full push-strong | 22/48 = 45.8% | 47/48 | 27/48 |
| P19 | second target-pull | 42/48 = 87.5% | 48/48 | 42/48 |
| P19 | second target-pull, half coefficient | 40/48 = 83.3% | 48/48 | 41/48 |

The final 50-image configuration was selected from these tested arms, with
P16's separately tuned configuration carried forward.

## Final 50-image confirmation

The strict proxy-success denominator is 300 per pair (six directions × 50
images); unconditional ASR uses all 500 images.

| Pair | Eligible TASR | Proxy success | Unconditional ASR |
|---|---:|---:|---:|
| P14 | 238/300 = **79.3%** | 300/300 | 238/500 = 47.6% |
| P16 | 162/300 = **54.0%** | 300/300 | 167/500 = 33.4% |
| P19 | 257/300 = **85.7%** | 300/300 | 260/500 = 52.0% |

The automatically selected final configurations were:

| Pair | CLS coefficient | Rep coefficient | Target weight | Source weight |
|---|---:|---:|---:|---:|
| P14 | 0 | 1.0 | 1.5 | 0.5 |
| P16 | 0 | 0.5 | 1.5 | 0.5 |
| P19 | 0 | 1.0 | 1.5 | 0.5 |

### Directional 50-image results

| Pair | G01 | G02 | G03 | G04 | G05 | G06 |
|---|---:|---:|---:|---:|---:|---:|
| P14 | 50/50 | 48/50 | 30/50 | 10/50 | 50/50 | 50/50 |
| P16 | 42/50 | 24/50 | 5/50 | 1/50 | 47/50 | 43/50 |
| P19 | 49/50 | 49/50 | 36/50 | 27/50 | 48/50 | 48/50 |

Butterfly-as-source directions are the hardest, especially butterfly→volcano
(G04) and butterfly→goldfish (G03). Volcano-as-source directions are much
stronger across all three pairs.

## Fixed-source and fixed-target views

Across all three pairs, aggregating the two outgoing directions per source:

| Source | CLS + pull/push | Pull/push-only |
|---|---:|---:|
| Goldfish | 40/48 = 83.3% | 42/48 = 87.5% |
| Butterfly | 19/48 = 39.6% | 14/48 = 29.2% |
| Volcano | 34/48 = 70.8% | 37/48 = 77.1% |

Aggregating the two incoming directions per target:

| Target | CLS + pull/push | Pull/push-only |
|---|---:|---:|
| Goldfish | 28/48 = 58.3% | 27/48 = 56.3% |
| Butterfly | 39/48 = 81.3% | 41/48 = 85.4% |
| Volcano | 26/48 = 54.2% | 25/48 = 52.1% |

Transfer depends strongly on direction, not only model family.

## Metrics and limitations

The primary metric is:

```text
eligible TASR = target targeted hits / proxy-success images
```

ASR measures images that leave the source class, whether or not they reach the
specified target. The stored diagnostics also include semantic-gap gain,
effective coefficients, proxy/target masks, and runtime.

The 50-image cohort is a clean-consensus cohort, not a random sample of all
ImageNet images. These results cover only intra-family small-to-large transfer.
P14/P19 selection is the best within the tested finite tuning grid, not a
global optimum.

## Prototype-distance analysis

For each proxy, source and target class prototypes were computed from the
48-image reference banks. The distance is `1 - cosine(source_prototype,
target_prototype)`. The resulting direction-level distances were joined to
the 50-image conditional TASR.

| Pair | Direction | Prototype distance | Eligible TASR |
|---|---|---:|---:|
| P14 | G01 | 0.2690 | 100% |
| P14 | G02 | 0.2374 | 96% |
| P14 | G03 | 0.2690 | 60% |
| P14 | G04 | 0.2744 | 20% |
| P14 | G05 | 0.2374 | 100% |
| P14 | G06 | 0.2744 | 100% |
| P16 | G01 | 0.2052 | 84% |
| P16 | G02 | 0.2777 | 48% |
| P16 | G03 | 0.2052 | 10% |
| P16 | G04 | 0.3282 | 2% |
| P16 | G05 | 0.2777 | 94% |
| P16 | G06 | 0.3282 | 86% |
| P19 | G01 | 0.0168 | 98% |
| P19 | G02 | 0.0182 | 98% |
| P19 | G03 | 0.0168 | 72% |
| P19 | G04 | 0.0247 | 54% |
| P19 | G05 | 0.0182 | 96% |
| P19 | G06 | 0.0247 | 96% |

The pooled 18-cell Spearman correlation between prototype distance and TASR is
approximately -0.27. Within-pair correlations are -0.25 (P14), 0.00 (P16),
and -0.37 (P19). Therefore prototype distance alone does not explain the
directional transfer differences in this experiment. The butterfly-source
failure pattern is likely related to directional decision geometry or class
conditional difficulty rather than a simple global prototype-distance rule.

As a secondary check, prototype distance correlates strongly with the measured
semantic-gap gain (Spearman 0.99 across the 18 cells), but not with TASR. This
means a larger source/target prototype separation makes the representation
objective move the embedding gap more, yet that movement does not reliably
cross the target model's class decision boundary. Representation movement and
targeted transfer are therefore distinct outcomes.

There is also a direct symmetry check. Prototype cosine distance is symmetric:
the distance for goldfish→butterfly equals butterfly→goldfish, and so on. The
measured transfer is not symmetric. For example, in P16 the same class-pair
distance appears for G01 and G03, but TASR is 84% for G01 and 10% for G03.
Likewise, G04 and G06 share the same P16 distance, while TASR is 2% versus
86%. This rules out a distance-only explanation and points to directional
source/target decision geometry.

## Evidence and reproduction

* [Experiment README](README.md)
* [Automated tuning and 50-image runner](run_auto_tune_then_50.sh)
* [Final selected 50-image config](config/threeclass_scale50_selected.yaml)
* [50-image results](../../outputs/threeclass_transfer_50/summaries/threeclass_scale50_results.csv)
* [8-image results](../../outputs/threeclass_transfer_8/summaries/results.csv)
* [Prototype distances](../../outputs/threeclass_transfer_50/diagnostics/prototype_distances.csv)
