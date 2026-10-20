# Does balanced Pull/Push generalize across compositions?

User request, 2026-10-20: expand composition ablation around0.5 Pull/0.5 Push.
Keep the completed diagnostic campaign unchanged. No final-target or commercial
feedback selects this study or its parameters; user confirmation is still needed
before any subsequent commercial testing.

Every row below is tested at **0.5/0.5 and0.75/0.25**, with all other settings
fixed: four depths25/50/75/100%, DI on, no margin/patch/CLS loss, spatial-token
mean pooling, equal per-image L1-normalized proxy gradients,50 iterations,
L∞16/255, step1/255, momentum1, random start, seed42, strict BF16 unquantized
model parameters,48 references/class. Attack uses surrogate image features only.

| Composition | Question |
|---|---|
| CLIP-L/14 | Vision-language contrastive single baseline |
| Qwen2B | Generative single baseline |
| InternVL2B | Generative single baseline |
| Gemma E2B | Generative single baseline |
| DINOv2-Large | Vision-only self-supervised single baseline |
| Qwen2B + InternVL2B | Remove CLIP from the original three-model ensemble |
| CLIP + Qwen2B + InternVL2B | Original ratio finding; reuse both completed rows |
| CLIP + Qwen2B + Gemma E2B | One VLM family replacement; reuse0.75 row |
| CLIP + InternVL2B + Gemma E2B | Third two-family VLM pairing |
| CLIP + DINO + Qwen2B | Replace one VLM with a vision backbone |
| CLIP + DINO + InternVL2B | Same proxy count, different VLM |
| CLIP + DINO + Gemma E2B | Same proxy count, different VLM |
| CLIP + DINO + Qwen2B + InternVL2B | Add DINO to the original ensemble |

This is13 compositions×two ratios =26 conditions, with three completed rows
reused and23 new attack conditions. Same ten historical development images
(five directions×two, positions2:4), and held-out Qwen9B, InternVL8B, Gemma E4B.
None of these checkpoints is an attack proxy. These are not family-disjoint
validators, untouched holdout images, or150-image final-target scores.

Report paired ratio wins/losses, per-validator and per-direction counts, matched
single/ensemble comparisons, optimization runtime and precision audits. Each row
contains30 dependent model-image observations on10 unique images. Resample by
image cluster for exploratory intervals; do not turn small development wins into
claims of robust superiority. No automatic test-set expansion or commercial calls.

Run `bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_pullpush_composition_ablation.sh`.
The single admission owner shares the global scheduler lock, caps total GPU
workers at4 and predicted reserved VRAM at72GiB, checks actual VRAM/host RAM,
and preserves checkpoints. It reuses the unchanged pure PP4 objective/worker via
an isolated audit/result routing adapter; all scientific sources are hash locked.

Outputs remain on persistent root disk under
`outputs/a100_precommercial_v7_targeted50/pull_push_composition_ablation/`.
Compact results publish under `reports/pull_push_composition_ablation/`.
