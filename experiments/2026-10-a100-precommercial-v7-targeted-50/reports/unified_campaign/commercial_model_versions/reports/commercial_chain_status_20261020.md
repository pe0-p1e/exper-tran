# Commercial model chain status — 2026-10-20

## Current state

Gemini and Claude CLI inference are currently blocked at the Antigravity account preflight. `agy models` returned `PERMISSION_DENIED (403)` with an account-level service eligibility error. This run sent no Gemini/Claude image inference requests after that denial. Previously recorded, trace-verified results remain available.

## Existing 10-image smoke results

| Requested model | Status | Target hits / 10 | ASR |
|---|---:|---:|---:|
| gpt-6.1-sol | COMPLETE | 4/10 | 0.4 |
| gpt-6-sol | COMPLETE | 8/10 | 0.8 |
| gpt-5.6-sol | BLOCKED | N/A/10 | N/A |
| gpt-6-astra | COMPLETE | 5/10 | 0.5 |
| gpt-6-luna | COMPLETE | 6/10 | 0.6 |
| gpt-5.6-terra | BLOCKED | N/A/10 | N/A |
| gpt-5.6-luna | BLOCKED | N/A/10 | N/A |
| gemini-3.8-flash-low | COMPLETE | 5/10 | 0.5 |
| gemini-3.8-flash-high | COMPLETE | 4/10 | 0.4 |
| gemini-3.8-flash-medium | COMPLETE | 4/10 | 0.4 |
| claude-sonnet-4-6 | COMPLETE | 8/10 | 0.8 |
| claude-opus-4-6-thinking | COMPLETE | 6/10 | 0.6 |

## Existing full 150-image results

Gemini’s three reasoning-profile runs already have complete local records for the fixed 150-image cohort (300 clean/adversarial requests each). These are reused results from the earlier campaign; the current parallel coordinator verified and aggregated them, it did not regenerate them.

| Requested model | Complete pairs | Target hits / 150 | ASR | Invalid replies |
|---|---:|---:|---:|---:|
| gemini-3.8-flash-low | 150/150 | 104/150 | 0.6933333333333334 | 0 |
| gemini-3.8-flash-medium | 150/150 | 102/150 | 0.68 | 0 |
| gemini-3.8-flash-high | 150/150 | 102/150 | 0.68 | 4 |

Direction counts (hits/30):

| Model | Goldfish→Butterfly | Pineapple→Guitar | Laptop→Espresso | Volcano→Rocking chair | Soccer ball→School bus |
|---|---:|---:|---:|---:|---:|
| gemini-3.8-flash-low | 21/30 | 20/30 | 20/30 | 30/30 | 13/30 |
| gemini-3.8-flash-medium | 21/30 | 20/30 | 19/30 | 30/30 | 12/30 |
| gemini-3.8-flash-high | 21/30 | 21/30 | 18/30 | 30/30 | 12/30 |

## Claude status

Claude’s existing 10-image smoke results are Sonnet 8/10 and Opus 6/10. The Sonnet 150-image folder has 15 fully paired images (102/300 requests); the remaining requests cannot be completed while the current Antigravity account eligibility error persists. No full-cohort Claude ASR is reported.

## GPT route status

On the same 10-image smoke cohort, completed models were gpt-6.1-sol 4/10, gpt-6-sol 8/10, gpt-6-astra 5/10, and gpt-6-luna 6/10. In the attempted 150-image stage, Codex returned a subscription usage-limit error on the first request; no retry or API billing fallback was used.

## Limits

These are fixed-label image-recognition transfer evaluations, not open-ended visual question answering. The 10-image checks are preliminary. CLI-reported model IDs are recorded, but backend serving snapshots and closed-model weight precision are not exposed. Full request and direction tables are in the adjacent JSON and CSV artifacts.
