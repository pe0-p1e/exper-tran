# Controlled proxy composition plan — 2026-10-19

Planning only. No new campaign has been launched. Existing frozen campaigns remain unchanged.

Per-family caps: CLIP <=4, DINO <=2, each eligible VLM family <=2. SigLIP2 uses one checkpoint in this plan. Entire target VLM family excluded.

C=CLIP L/14; C1=B/32; C2=B/16; C4=L/14@336; D=DINOv2 Large; D1=DINOv2 Base; S=SigLIP2 So400m/14-384. A1/A2 and B1/B2 are the two eligible VLM families at their small/medium sizes.

| Target | A1 / A2 | B1 / B2 |
|---|---|---|
| Qwen3.5-27B | InternVL3.5-2B-HF / InternVL3.5-4B-HF | Gemma 4 E2B-it / Gemma 4 E4B-it |
| InternVL3.5-14B-HF | Qwen3.5-2B / Qwen3.5-4B | Gemma 4 E2B-it / Gemma 4 E4B-it |
| Gemma 4 31B-it | Qwen3.5-2B / Qwen3.5-4B | InternVL3.5-2B-HF / InternVL3.5-4B-HF |

| ID | Tier | Composition | K | Existing complete target rows |
|---|---|---|---:|---:|
| M01 | core | C | 1 | 2/3 |
| M02 | core | A1 | 1 | 3/3 |
| M03 | core | B1 | 1 | 2/3 |
| M04 | core | C + A1 | 2 | 3/3 |
| M05 | core | C + B1 | 2 | 3/3 |
| M06 | core | A1 + B1 | 2 | 3/3 |
| M07 | core | C + A1 + B1 | 3 | 3/3 |
| M08 | core | C + D | 2 | 0/3 |
| M09 | core | C + S | 2 | 0/3 |
| M10 | core | D + S | 2 | 0/3 |
| M11 | core | A1 + A2 | 2 | 2/3 |
| M12 | core | B1 + B2 | 2 | 0/3 |
| M13 | core | C + D + A1 | 3 | 0/3 |
| M14 | core | C + D + B1 | 3 | 0/3 |
| M15 | core | C + D + A1 + B1 | 4 | 1/3 |
| M16 | core | C + A1 + A2 | 3 | 0/3 |
| M17 | core | C + B1 + B2 | 3 | 0/3 |
| M18 | core | A1 + A2 + B1 + B2 | 4 | 0/3 |
| M19 | extension | D | 1 | 0/3 |
| M20 | extension | S | 1 | 0/3 |
| M21 | extension | C2 + C | 2 | 0/3 |
| M22 | extension | D1 + D | 2 | 0/3 |
| M23 | extension | C1 + C2 + C + C4 | 4 | 0/3 |
| M24 | extension | C2 + C + D1 + D | 4 | 0/3 |
| M25 | extension | S + D + A1 | 3 | 0/3 |
| M26 | extension | S + D + B1 | 3 | 0/3 |
| M27 | extension | C + S + A1 + B1 | 4 | 1/3 |
| M28 | extension | D + S + A1 + B1 | 4 | 0/3 |
| M29 | extension | C2 + C + A1 + B1 | 4 | 0/3 |
| M30 | extension | D1 + D + A1 + B1 | 4 | 0/3 |
| M31 | extension | C + D + A1 + A2 + B1 + B2 | 6 | 0/3 |
| M32 | extension | C1 + C2 + C + C4 + D1 + D + A1 + B1 | 8 | 0/3 |

## Controlled contrasts

- Visual family identity at K=2: M08/M09/M10/M21/M22. Different architectures/native resolutions remain part of checkpoint identity; this is not a pure training-objective causal test.
- Fixed C, compare same-family/cross-family VLM pairs at K=3: M07/M16/M17.
- Fixed C+D, add A1 vs B1 at K=3: M13/M14; add both at K=4: M15.
- Fixed A1+B1, compare visual pairs at K=4: M15/M27/M28/M29/M30.
- Fixed total K=4, vary VE/VLM allocation: M23 (4+0), M15 (2+2), M18 (0+4). This changes identities as well as allocation; replicate across families and do not attribute solely to ratio.
- Grow family multiplicity: M06 -> M11/M12 -> M18, and M07 -> M16/M17 -> M31.
- Large saturation test M32; report runtime and actual forward/backward count, not ASR alone.

## Reuse and validity

CSV status REUSE_CANDIDATE means exact target+proxy identity matches a completed exported A3 measurement; reuse is finalized only after model revisions, input/reference manifests, layer/pooling, loss, patch weight .35, DI, fusion order/RNG semantics, 50 steps, seed, BF16, PNG hashes and raw evaluation records all match. Preserve original proxy execution order for reused results. Never treat V6 Pull/Push as A3. RESUME_EXISTING means inspect the existing owner/checkpoint and resume rather than duplicate it. Do not trust an older pending selection table over actual supplement artifacts.

All primary comparisons use the same 5x30 ground-truth cohort, 48 refs/class, four normalized depths, A3, equal L1 fusion, eps16/255, step1/255, momentum1, random start, seed42 and strict BF16. Target is post-freeze only. Hold loss/layers/fusion fixed for this composition study; running loss ablations are separate.

New candidates require matched single-member baselines for any strongest-single/ensemble-gain claim. Prefer already generated identical target-free PNGs and post-freeze evaluation on another target when proxy identities, settings and target-free cache contracts are exactly identical; do not assume matching image IDs alone suffices.

## Estimated elapsed execution (one A100 80GB)

Measured completed 150-image attack loops: CLIP single ~4.5min; small-VLM singles ~4.6-23.8min; two-proxy ~16-68min; three-proxy ~15-63min; four-proxy ~67-71min. These are observed under changing contention/microbatches and exclude some loading/evaluation/analysis overhead. They are not directly additive GPU throughput predictions.

Core additional work including audits/calibration, missing matched singles and evaluation: planning envelope 18-30h. Full core+extension additional work: 40-70h, including small development checks but excluding commercial/generalization, current-campaign remainder and multi-seed confirmations. Large/new model throughput is unmeasured; offload or expensive native-resolution backward can exceed this envelope. Re-estimate after first full representative batch of each new model. Four workers are an upper bound, not a 4x speed assumption.

Do not expand every new template immediately: first independent 10-image surrogate-dev screening, freeze a small predeclared confirmatory subset without final-target feedback; then complete formal five-direction rows. Full 96-row matrix is a candidate budget, not a promise that every row will be launched.

Counts at planning time: {"core": {"REUSE_CANDIDATE": 22, "RESUME_EXISTING": 3, "NEW_PLANNED": 29}, "extension": {"NEW_PLANNED": 35, "RESUME_EXISTING": 6, "REUSE_CANDIDATE": 1}}

Current primary campaign: 22/32 attacks and 8/32 evaluations recorded, zero recorded failures. Original table 11/13 complete; E08 queued; E09 112/150 attacks and no evaluations. E09 uses three same-family VLMs, outside the newly capped two-per-family future design; preserve prior data and mark this historical row deferred, not a mandatory new-design task.
