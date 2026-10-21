# Final experiment closeout

Updated UTC: 2026-10-21 16:43:38

GPT50 is USED FOR SELECTION by explicit user instruction. Final150 is separate; final outcomes cannot change frozen settings.
Models: GPT-6.1 Sol, GPT-6 Sol, GPT-5.6 Sol, all low; no Astra, no paid API fallback.
Pure PP, BF16 unquantized,50steps,16/255,DI,equal per-image L1-normalized proxy gradients. References48/class. Ratios are separate Uniform4 controls.

## Actual completed/partial open-weight evaluations

|Condition|Target|Hits / N|Complete|DeltaR|CKA|RSA|
|---|---|---|---|---|---|---|

## commercial50: PARTIAL
- gpt-6.1-sol / selected_prior: 35 hits, 50/50 completed; ASR=0.7
- gpt-6.1-sol / uniform4: 35 hits, 50/50 completed; ASR=0.7

## commercial150: PARTIAL

## ratio30: PARTIAL
- gpt-6.1-sol / pull_0.5: 19 hits, 30/30 completed; ASR=0.6333333333333333

Missing measurements remain N/A, never zero. Closed-model target DeltaR/CKA/RSA are unavailable. Correlations are exploratory and correlated across shared images/models. No causality claim.
PCA/t-SNE are fitted jointly per model/class pair; missing reverse-direction attacks are explicitly absent, not fabricated. No new reverse attack campaign is claimed.
Complete numerical receipts, gradient traces, configurations and image-level results are in the uploaded compressed data bundle; full PNGs and raw CLI traces stay on the persistent root disk.
