# Method and code cross-check, 2026-10-20

Scope: relevant methods, selection protocols, ensemble definitions and evaluation sections; not a proof audit or an exhaustive literature review. Full downloaded papers/code stay on the persistent local disk under `outputs/a100_precommercial_v7_targeted50/literature_review/`. No new ILA/FDA implementation or experiment result is claimed here.

## FDA(N): direct joint combination remains future work in THIS paper

Main paper §3/§4.1 and supplement A–C checked. The objective averages target-class auxiliary probabilities plus relative feature disruption. Optional output cross-entropy is a distinct variant. Supplement B selects the best singleton on200 held-out images, then sweeps each added layer conditional on the current subset, continuing to five layers. Supplement C explicitly distinguishes single-source multi-layer FDA from the multi-source output-based ensemble comparator and leaves their combination to future work. Auxiliary predictors are Conv–ReLU–Pool–Conv–ReLU–Pool–Dropout–Linear, trained8,000 steps. Thus our prototype-only approach is an adaptation, not a reproduction. Candidate-level native block identities and real conditional subset gains matter more than uniform depth percentages. FDA(N) is empirical methodological support, not a theorem proving universal multilayer superiority.

Sources: https://proceedings.neurips.cc/paper/2020/file/eefc7bfe8fd6e2c8c01aa6ca7b1aab1a-Paper.pdf ; https://proceedings.neurips.cc/paper/2020/file/eefc7bfe8fd6e2c8c01aa6ca7b1aab1a-Supplemental.pdf

## MFAA: multi-layer importance, but mask ensemble is not model count

Publisher abstract and the official455-line attack entry point checked; full paper unavailable through the accessed publisher route. Code computes class-logit gradients at several feature taps, propagates guidance between layers and attacks importance-weighted features. `ens` counts random masked inputs for attention estimation. The entry point builds one named network, not several checkpoints. Publisher reports model-ensemble experiments, but their precise configuration remains unverified. Therefore do not equate ensemble attention with heterogeneous proxy fusion or claim the public script establishes that equivalence. Its logit-derived importance also differs from our output-head-free prototype objective.

Sources: https://ieeexplore.ieee.org/document/10833658/ ; https://github.com/KWPCCC/MFAA ; pinned source c8174848f0170b2719def9ba2a2eedf7a0d0041e, `Multi-Feature Attention Attack.py`, particularly lines261–322 and391–447.

## FTM: genuine nearby joint work, different objective and mechanism

§3.3, §4.2–4.3, appendix A.2–A.4 and official attack code checked. Feature perturbations oppose targeted optimization while the image is optimized to withstand them; only a random subset of layer perturbations is updated each iteration. Core FTM-E uses independently perturbed copies of one backbone; separate experiments also combine RN50+Inception and RN50+LeViT. Thus heterogeneous model plus multilayer interventions already exist. It uses output-logit attack loss rather than direct prototype movement. MLLM appendix asks target-label-conditioned yes/no questions, unlike our fixed ten-class prompt; reported rates are not directly comparable. FTM is important nearest prior art, not evidence that our representation-only layer/model selection has already been solved.

Sources: https://arxiv.org/html/2411.15553v2 ; official https://github.com/uhiu/feature-tuning-mixup at912b8da37288ad1c339bf974408ca20d4e9bdf3d, `attacks.py` and `main.py`.

## ICML2025 ensemble theory: assumptions matter

Definitions3.1/3.3, Theorem4.1, Lemma4.2 and Theorem4.3 assumptions checked against the conference PDF. Transferability error is a population-loss optimality gap, not1−targeted-ASR. Diversity is prediction variance; the squared-error decomposition has a KL extension. Complexity bounds impose stated boundedness/Lipschitz/dependence conditions. These results motivate strength–diversity tradeoffs, but do not identify lowCKA, orthogonal pixel gradients, our cosine prototype objective, or correlated layers of one encoder with the theorem's quantities. Applying it to model-layer components requires a new justified derivation and tests; no automatic guarantee for fewer components or universal coverage follows.

Sources: https://proceedings.mlr.press/v267/yao25f.html ; conference36-page PDF https://raw.githubusercontent.com/mlresearch/v267/main/assets/yao25f/yao25f.pdf ; readable version https://arxiv.org/html/2410.06851v3 .

## Consequences for our implementation

1. Implement ILA's actual baseline displacement projection and layer-specific propagation; the existing scan is not that experiment.
2. Implement conditional layer-subset validation inspired by FDA, with model-specific block identities. Label prototype replacement explicitly and keep a second pair branch to test a greedy failure.
3. Preserve our simple visual-only loss when testing selection; FTM/MFAA are distinct mechanisms, not silent additions to the baseline.
4. Treat model diversity, useful directed agreement, redundancy and cost as separate quantities. Test joint layer×model effects with matched conditions and independent confirmation.
5. No first-ever multi-model/multi-layer claim, no universal-head claim and no theory-supported-ASR guarantee. Existing pilot findings remain exploratory.
