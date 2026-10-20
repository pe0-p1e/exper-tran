# Model–layer pairwise control

Six development-selected atomic units per validator produce 15 real joint attacks: three within-model layer pairs and twelve cross-model pairs. No union of singleton attack images is treated as an ensemble.

I = T(pair) − T(left) − T(right) + T(clean). All four scores use the same frozen image IDs, labels, validator, prompt and unconditional denominator. Cached clean responses must agree. This additive interaction on a bounded binary metric can be negative even when the pair beats both members (ceiling effects); it is not a standalone selection score or proof of semantic complementarity. Also inspect joint-only hits and gains/losses against the stronger singleton.

Uncertainty resamples paired images within directions. These are small, development-selected comparisons, not untouched confirmatory significance tests. Missing/partial results are not zeros. Within-model two-layer pairs and cross-model pairs differ in backward cost; report measured runtime, calls and peak memory before attributing gains to complementarity. Timing may depend on concurrent workloads. No new loss, inference or commercial feedback is introduced by this analysis.
