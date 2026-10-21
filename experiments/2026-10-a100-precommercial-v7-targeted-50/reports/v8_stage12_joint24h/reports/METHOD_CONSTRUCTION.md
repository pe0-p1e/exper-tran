# Integrated method construction — Stage1

Purpose: construct one source-guided, compute-aware model–layer selection method. Full component ablations are deferred to Stage3. Existing matched experiments remain reusable evidence, with their original configuration and model order.

The primary image attack is unchanged:50-step1:1 cosine prototype Pull/Push,48 references/class, strict BF16 parameters, DI and equal L1-normalized proxy gradients. No language-head loss or final-target feedback.

ILA is a separately costed source-only diagnostic. Starting from clean, maximize projection along a frozen50-step PP baseline feature displacement for10 sign steps, then inspect relative displacement propagation through every native block on saved/reloaded PNGs. Use raw valid spatial tokens. Record the deepest interior self-peak, with explicit NO_PEAK rather than silently substituting a maximum. ViT preprocessing, source PP baseline and hyperparameters make this an adaptation, not a published benchmark reproduction. Extra diagnostic images are never scored as primary PP attacks.

Candidate generation combines the ILA source proposal with positive one-step PP effectiveness. At identical random-start/DI states, compute image-gradient cosine/negative-dot fractions and CKA/RSA on aligned480 reference images. A bounded source heuristic screens TWO additions per beam parent: one resident-model layer and one new-model unit when both exist. CKA novelty, gradient residual novelty, conflict and predicted marginal cost guide screening; none alone proves useful complementarity.

FDA-inspired conditional search generates actual50-step joint PP attacks for these candidates and evaluates only held-out development validators. A two-state quality/compute Pareto beam grows model–layer subsets. Conditional gains count gained AND lost images relative to the parent. This is a bounded surrogate shortlist adaptation, not exhaustive FDA(N) or trained FDA auxiliary predictors. A nonzero marginal-cost floor prevents treating extra resident taps as free. Concurrency affects timings.

Uniform and at most two integrated finalists advance to independent30-image exploration. The selected and uniform methods freeze before50-image confirmation and subscription GPT testing. A5pp development tolerance is not equivalence evidence. Failed improvement and missing results remain explicit. Commercial responses never retune the frozen selection.

Sources: ILA algorithm2,§3.1,§4.2 https://arxiv.org/html/1907.10823v3 ; FDA(N) supplementaryB https://proceedings.neurips.cc/paper/2020/file/eefc7bfe8fd6e2c8c01aa6ca7b1aab1a-Supplemental.pdf . The geometric shortlist and joint Pareto rule are our experimental heuristics, not theorems or faithful implementations claimed from these papers.
