# Layer and model selection: active small pilot

Started 2026-10-20 UTC. Launcher: `run_layer_selection_pilot.sh`.
Permanent output root: `/home/ubuntu/exper-tran/outputs/a100_precommercial_v7_targeted50/layer_selection_pilot/`.

The preceding source-only scan completed five models and 115 native visual blocks with no failures. These descriptive geometry measurements are not attack ASR. They do not yet establish a layer-selection rule.

The next pilot fixes historical dev10 (two images in each of five directions), 48 references/class, 50 steps, epsilon16/255, step1/255, momentum1, seed42, BF16, balanced .5/.5 Pull/Push, DI and independently L1-normalized equal proxy fusion. No final target inference and no commercial evaluation. Qwen9 is a held-out development surrogate; all attack proxies exclude the entire Qwen family.

Six layer conditions use the identical E11 order InternVL2B + GemmaE2B + CLIP-L/14:

| Condition | Normalized visual depth(s) |
|---|---|
| Single non-quartile | 37.5% |
| Single non-quartile | 62.5% |
| Single non-quartile | 87.5% |
| Single prior representative | 75% |
| Two layers | 62.5%,87.5% |
| Four-layer matched baseline | 25%,50%,75%,100% |

Four additional conditions keep the four uniform layers and compare InternVL alone, Gemma alone, CLIP alone and InternVL+Gemma. Together with the three-model baseline these support matched model-count comparisons. Exact native block indices must be read from saved model/attack metadata, not inferred from percentages alone. Valid matching baseline attacks are reused; four-uniform Qwen9 validation is reused only after cohort, scientific-source and PNG-hash checks.

The scheduler uses the shared admission lock, at most four workers and a72GiB aggregate reservation budget. A two-step smoke precedes new attacks. Checkpointed PNGs are evaluated only after freeze. Failures remain explicit rather than scored as zeros.

Expected first comparisons:20–30min; full bounded pilot:45–90min based on prior E11 attacks taking roughly4–6min/condition, plus model loading/reference preparation and validation. This is an estimate, not a completion guarantee.

Interpretation: small exploratory data already used in development, not an untouched confirmation set. Compare paired image successes, per-direction counts, runtime and memory. Non-quartile taps are predeclared exploratory controls, not an implementation of a literature selection rule. A later independently validated rule should outperform or simplify these controls before formal evaluation.

## Literature to cite and limits of the adaptation

ILA: Huang et al., *Enhancing Adversarial Example Transferability with an Intermediate Level Attack*, ICCV2019, pp4733–4742. [Paper](https://arxiv.org/abs/1907.10823), [official implementation](https://github.com/CUAI/Intermediate-Level-Attack).

ILA refines a baseline adversarial feature displacement, with a projection objective in its intermediate representation. Its source-only layer-selection heuristic examines relative feature disturbance after layer-specific refinement and chooses the latest layer still producing a local peak. Our all-layer baseline displacement measurements do **not** reproduce this rule: they lack the layer-specific ILA refinements. Its classifier transfer observations motivate testing intermediate layers, but do not prove a universal optimum for targeted VLM transfer.

FDA(N): Inkawhich et al., *Perturbing Across the Feature Hierarchy to Improve Standard and Strict Blackbox Attack Transferability*, NeurIPS2020. [Paper](https://arxiv.org/abs/2004.14861).

FDA(N) combines class-distribution objectives over multiple layers using learned auxiliary classifiers and feature disruption. It chooses layer sets with iterative greedy held-out transfer validation. The optional output cross-entropy variant differs from our visual-only objective. We borrow the question of which layer subsets transfer and the idea of held-out surrogate validation, not its auxiliary classifier, final-target feedback, or reported success rates. More layers/models are not guaranteed to help.

Reading status: core methods and relevant layer-selection/results sections have been checked; full appendix/official-code cross-check remains unfinished. No claim of a faithful ILA/FDA reproduction. Further reading notes will record the exact implementation differences before any literature-derived selection algorithm is claimed.
