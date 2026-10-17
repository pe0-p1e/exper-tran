# V7 targeted transfer optimization (>=50% each target and each direction)

This branch was opened after the user explicitly raised the requirement from the V6 engineering gate to >=50% target targeted ASR for **every target and every directed class pair**. There is **no claim** that such rates have already been achieved.

The existing V6 BF16 recovery baseline is complete: 65/65 cells, 15/15 all-parameter BF16 checks, three target best pooled ASRs Qwen27B 13.33%, InternVL14B 14.00%, Gemma31B 1.33%. These are low and some pair directions remain zero. V6 records are **historical and immutable**.

Start here:
1. `AGENTS.md` — science protocol and implementation priorities;
2. `experiment_plan.yaml` — exact acceptance metrics and experiment arms;
3. `../2026-10-a100-precommercial-v6/MULTIPLE_PROXY_RECOVERY.md` — correct no-clean-classifier gate and V6 artifacts;
4. `../2026-10-a100-precommercial-v6/run_multiple_proxy_recovery.sh` and `../2026-10-a100-precommercial-v6/src/` — working BF16 attack/eval code to reuse.

## Key requirement

A final selected method must achieve >=15 successful targeted transfers among the same 30 images for **each** of T01,T03,T05,T07,T09, separately for EACH Qwen3.5-27B, Gemma4-31B-it, InternVL3.5-14B-HF (15 cells). Prefer >=16/30. This is a **goal**, not a fabricated guarantee.

Every method must use only surrogate intermediate image representations at attack generation, target evaluation strictly post-freeze, locked image cohorts, strict BF16 weights, and explicit perturbation-budget declarations.

## Implementation order

0. Run data/precision/checkpoint/visual-embedding-gradient diagnostic, inspect failure directions and Qwen/Gemma geometry scales.
1. Establish stronger single-proxy baselines on the exact fixed cohort, and use separate surrogate development examples for choosing methods.
2. Implement representation-margin, multi-layer, and patch-local/global loss arms independently; preserve original 50-step equal-weight V6 baseline.
3. Implement surrogate-only adaptive proxy selection and conflict-aware gradient fusion, with controlled ablations.
4. Implement differentiable input diversity and gradient variance/smoothing; separately benchmark 100/200-step and multiple-start higher compute variants, with epsilon unchanged.
5. Fix candidate algorithms BEFORE accessing target outputs; evaluate frozen PNGs and audit exact per-target/per-class results.
6. Verify seeds 42/43/44 and make one aggregate scientific report; if any target/direction <50%, document exactly what remains and why.

## Expected single entry point to implement

```bash
bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_all_v7.sh
```

Do not claim this launcher exists until Codex has implemented it. Scripts must support resumption and clear progress. If no remote A100 access is available in Codex environment, stop with exact blocker and instructions rather than marking the experiment complete.

Output root: `outputs/a100_precommercial_v7_targeted50/`.

No change to original spreadsheet layout; result exports may use a separate V7 CSV and must preserve exact denominators.
