# Identify the component responsible for improvement

User override, 2026-10-20: **do not rerun the full multi-proxy matrix**. Test one
or two representative combinations, explain the decisive changes, and keep the
method simple. No new commercial requests until the user approves an exact
proxy list and setting. Historical Margin and V6 results remain untouched.

Two fixed combinations, chosen before this diagnostic evaluation:

- A: CLIP ViT-L/14 + Qwen3.5-2B + InternVL3.5-2B-HF.
- B: CLIP ViT-L/14 + Qwen3.5-2B + Gemma 4 E2B-it.

They differ by one proxy family. Both use held-out Qwen3.5-9B,
InternVL3.5-8B-HF and Gemma 4 E4B-it for development validation; none is an attack
proxy. These are held-out model checkpoints, not disjoint architectural families.

| Condition, on each combination | Global loss | Depths | DI | Patch |
|---|---|---|---|---|
| deep_PP_noDI | 0.75 Pull /0.25 Push | 100% | off | 0 |
| four_PP_noDI | same | 25/50/75/100% | off | 0 |
| four_PP_DI | same | four | on | 0 |
| four_margin_DI | prototype margin | four | on | 0 |
| four_margin_patch_DI | prototype margin | four | on | 0.35 |

Combination A also tests pure four-layer Pull/Push ratios 1/0, 0.75/0.25,
0.5/0.5, 0.25/0.75, 0/1, holding all other settings fixed. The 0.75 row is shared.
There are14 unique conditions. Two compatible A Margin conditions reuse existing
U02/U00 measurements after cohort, model, loss, layers, DI and PNG validation.
Thus12 development attack conditions are new, plus an isolated2-step GPU smoke.

Each condition uses the same10 images, five directions×two images, from
historical development positions2:4. They are disjoint from the commercial10,
the final150, and48 references/class; this is not an untouched population holdout.
Report all30 model-image observations, per-model/per-direction outputs, and
paired wins/losses. Bootstrap by the10 unique images, retaining all three
validators within each image. Small differences are inconclusive; no equivalence
claim, p-value shopping, or automatic expansion to150 images/96 combinations.

Attack weights are BF16, unquantized; losses/gradients are FP32 as appropriate.
L∞16/255,50 steps, step1/255, momentum1, random start, seed42,224px,
48 references/class, spatial-token mean pooling. Loss is calculated separately
at each layer in each proxy. Four depths are averaged uniformly. Each proxy's
per-image pixel gradient is L1-normalized before equal averaging. No CLS loss,
output-head gradients, or final-target feedback.

One GPU admission owner adopts the previous controller's active detached
workers and lets them finish. Previous pending Margin expansions are paused.
New work observes at most4 total GPU processes, a72GiB predictive reservation
budget, actual available memory, and12GiB minimum host-RAM headroom. Per-process
allocation caps remain enabled. Failures remain explicit; no precision fallback.

Run: `bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_pullpush_diagnostics.sh`.
Initial transfer of ownership adds `--takeover`; it signals only the previous
controller, preserving detached workers and all artifacts. Later resumes omit it.

Outputs: `outputs/a100_precommercial_v7_targeted50/pull_push_four_layer/` on
the persistent root disk. Read `reports/conditions.csv`, `per_image.csv`,
`paired_components.csv`, `report.md`, BF16 audits and exact `protocol_lock.json`.
`commercial_approval.json` starts with `approved:false`; this launcher contains
no commercial adapter or final-target evaluation stage. Candidate selection
and commercial testing require a subsequent explicit user decision.
