# Selection objective: complementary coverage per unit compute

User correction, 2026-10-20: individually strongest layers need not form the best layer subset; the same applies to model subsets. The active ten-condition pilot is a set of controls, not a sufficient selection algorithm. Preserve its frozen protocol and completed measurements.

## Scientific question

Find a small collection of model-layer objectives that transfers across held-out surrogate representations and class directions under a measured computation budget. Universal representation coverage is a hypothesis to operationalize, not an already established universal visual space or a claim of aligned embeddings across architectures.

Candidates are (checkpoint revision, native visual block). Each retains its own source/target prototype space. Shared pixel-space perturbations provide a common domain for gradient comparisons; unequal-dimensional embeddings must not be concatenated or treated as directly comparable coordinates.

## Measurements

- Representation redundancy: paired-image linear CKA and RSA on aligned image IDs; descriptive, not a selection proof.
- Directional compatibility: signed gradient cosine, negative-dot fraction, and norm of fused normalized gradients relative to summed component norms, computed at identical image/DI states. Low similarity alone is not desirable: a novel direction may be noise or oppose every useful component.
- Useful novelty: for component i, measure surrogate objective improvement under its projected update on other component spaces. Report signed improvement and harmful interference, rather than rewarding orthogonality unconditionally. Analysis-only cross-evaluation must not alter the attack update or consume final-target feedback.
- Conditional coverage: on held-out development surrogates, measure the actual subset attack's gains on images and class directions missed by the existing subset, plus regressions on previously successful cases. Do not replace subset optimization by the union of independently successful attacks; each subset must produce a single valid PNG per input.
- Cost: measured walltime, visual forwards/backwards, peak memory and model reload/offload overhead. Adding a tap to a resident visual forward has different marginal cost from adding a model. Also measure backward graph/activation costs; layer count is not an adequate compute proxy.

## Bounded search after current controls

1. Use the five-model all-native-layer scan to screen redundant regions and propose a small candidate pool without final-target feedback. Keep an unranked diversity control so the screen cannot exclude all individually weaker complementary taps.
2. Compare same-cardinality subsets: strongest-single-based pairs; complementary candidate pairs; and depth-spread controls. Include pairs whose individual members are weaker. Search at least a small beam of alternative subsets; greedy top-single addition can miss jointly useful pairs and is not assumed optimal.
3. Keep model and layer factors separate initially, then cross the most informative small subsets. Report whether complementarity depends on the model as well as depth. No automatic large Cartesian sweep.
4. At each stage report marginal benefit, harms and cost relative to the current subset. Retain the empirical Pareto frontier of held-out surrogate directional performance versus measured compute. No claim of submodularity, guaranteed greedy optimality, or universal monotonic improvement.
5. Freeze finalists before confirmation. Historical dev10 is exploratory; independent confirmation image IDs must be audited against every prior study before being called untouched. Hold out additional surrogate architectures where feasible. A single Qwen9 validator cannot support a universal-transfer claim.
6. Prefer the cheapest subset meeting a predeclared directional performance tolerance on confirmation, with uncertainty reported; do not choose a statistical tolerance after seeing confirmation scores. Define that tolerance in the next immutable experiment protocol before launching the confirmation.

All attacks remain strict BF16, balanced representation Pull/Push, fixed epsilon/steps/DI/normalization unless explicitly isolated in a separate ablation. Final targets and commercial outputs are evaluation-only; new commercial proxy settings still require the user's explicit confirmation.

## Status

Current all-layer scan: complete (115 model-layer pairs). Active layer/model pilot: running. Complementarity-guided subset search and independent confirmation: next-stage design, not yet executed. ILA and FDA(N) inform layer sensitivity and subset validation; neither establishes that the individually strongest layers must combine optimally.
