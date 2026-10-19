# Corrected strict cross-family campaign

User clarification, 2026-10-19: keep the existing Gemma31B results because its
proxies contain no Gemma. Redo Qwen and InternVL excluding their complete model
families. The task remains fixed ten-class image recognition, not open-ended VQA.

The default launcher and `run_family_disjoint_recovery.sh` execute this correction.
All new outputs use `outputs/a100_precommercial_v7_targeted50/family_disjoint_recovery/`.
Previously published numerical measurements and Gemma artifacts are preserved.
Older Qwen/InternVL results must be described as family-overlap transfer.

## Frozen combinations and scope

- Qwen27B: CLIP ViT-L/14 + InternVL2B + Gemma E2B; no Qwen proxy of any size.
- InternVL14B: CLIP ViT-L/14 + Qwen2B + Gemma E2B; no InternVL proxy of any size.
- Same main algorithm: four-layer class-prototype margin, local patch weight0.35,
  DI, equal per-image L1-normalized gradients, 50 steps, epsilon16/255,
  alpha1/255, momentum1, random start, seed42, 224px canvas, strict BF16.
- Keep the five original GT-selected directions, 30 test images/direction,
  48 disjoint references/class, original prompt and parser. No clean-output gate.

Each target has 16 fixed conditions: primary multi; every one of its three proxy
members individually; layers1/25/50/75/100%; no patch; patch-only; soft-reference
aggregation; original deepest-layer Pull/Push without DI; no DI; and primary
seeds43/44. Thus 32 conditions x5 directions =160 cells =4,800 image evaluations.
Primary seed42 remains the main result regardless of other seed results.
The legacy Pull/Push condition changes multiple components and must not be
presented as a single-factor ablation. No-patch/no-DI are matched component tests.

The entire matrix is locked before corrected target evaluation. The correction
uses the earlier selected algorithm and a fixed Gemma E2B replacement, not
another search on known target outputs. Prior target results were already seen:
disclose this is protocol correction on an existing test cohort, not an untouched
new benchmark. Do not retune after any corrected target result is observed.

## Enforcement and execution

The attack CLI checks `--excluded-proxy-family` before loading any model.
The evaluator independently checks the predeclared condition/target association
and actual state-file proxy list before loading the target. Every BF16 load is
audited. Two-image/two-step gradient/PNG smokes precede scientific generation.
Both primary ensembles and matched singles run before the remaining predeclared
ablations. Target evaluations may be reported while the remaining fixed queue
continues; they cannot alter that queue. Resume reuses valid PNG batches/results.

Final checks cover family isolation, exact image IDs, PNG hashes, finite traces,
all50 steps, L-infinity, BF16 and numerator/denominator. The report includes
per-direction results, all three matched singles, paired bootstrap differences,
seed results, compute/runtime and failures. Primary-only target geometry includes
DeltaR, variance, margins, CKA/RSA and joint PCA/t-SNE, in a new isolated analysis
directory. No reverse-direction claims are possible without reverse cohorts.

Everything stays on persistent root disk. Upload compact CSV/JSON reports/audits
and a few real examples/plots; exclude the full PNG corpus and embedding caches.
