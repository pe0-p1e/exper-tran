# V8 Stage 1–2 handoff

Status: RUNNING. Completed evaluations: 138/150 planned so far.

Algorithm: strict BF16, native spatial-token mean features,48 references/class, .5 cosine target pull + .5 source push,50 steps,L∞16/255,DI, equal per-image L1-normalized proxy gradients. Layer losses are averaged inside each model before proxy normalization. No output-head, patch or margin loss.

Family exclusion applies to each actual validator, including Qwen9/InternVL8/GemmaE4. Final open-weight target150-image cohorts remain sealed. Historical10-image screening is development data; exploratory30 and confirmation50 are separately frozen. Stage2 reuses confirmation images only after selection lock.

BF16 audit load events: 428. Failed audit events: 0.

Failures retained: 0. See scheduler.json and attempts/*/exit.json; missing results are not zeros.

## Frozen methods
PENDING: no selected method or superiority claim.

## GPT confirmation
PENDING: no calls/results claimed.

## Interpretation and Stage3 discussion
No claim of universal visual head, proven causal mechanism or first multi-model/multi-layer attack. CKA/RSA and gradient conflict are diagnostics, not success surrogates. An exact implementation combination alone is not a novelty claim.

Must discuss: held-out categories and COCO original benchmark protocols; faithful vs adapted FDA/FOA/FTM/AdaEA/SEA baselines; multi-seed42/43/44; JPEG/resizing robustness; ten-class targeted ASR vs semantic change; nested surrogate selection and model family seen during development; independent final targets; selection cost and amortization.

All observed results: reports/conditions.csv,per_image.csv,directions.csv. Exact methods and hypotheses remain pending until measurements complete. No paper conclusions are fabricated.
