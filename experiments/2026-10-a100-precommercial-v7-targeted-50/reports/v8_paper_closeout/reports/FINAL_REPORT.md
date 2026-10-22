# Final experiment closeout

Updated UTC: 2026-10-22 02:54:17

GPT50 is USED FOR SELECTION by explicit user instruction. Final150 is separate; final outcomes cannot change frozen settings.
Models: GPT-6.1 Sol, GPT-6 Sol, GPT-5.6 Sol, all low; no Astra, no paid API fallback.
Pure PP, BF16 unquantized,50steps,16/255,DI,equal per-image L1-normalized proxy gradients. References48/class. Ratios are separate Uniform4 controls.

## Actual completed/partial open-weight evaluations

|Condition|Target|Hits / N|Complete|DeltaR|CKA|RSA|
|---|---|---|---|---|---|---|
|V8_a1cfea7510e1912df1|Qwen3.5-27B|137/150|True|0.0003372924510525384|0.5308645497187259|0.3736688014319241|
|V8_3a38a4494a7cd9543c|Qwen3.5-27B|137/150|True|0.00033191460903860085|0.4875657072376558|0.3479861367182147|
|V8_adddcc5924bd7a7ca4|Gemma 4 31B-it|134/150|True|0.019495939754415305|0.5397599036752193|0.5012941824816917|
|V8_22525cf8872c09b324|Qwen3.5-27B|138/150|True|0.0003310475494678637|0.5221046832863129|0.37372678215397515|
|V8_4022eb00d5f4bee0e4|Gemma 4 31B-it|139/150|True|0.019276889200943212|0.5692165732516011|0.5177944410730784|
|V8_57d6dc9f834c2edb3b|InternVL3.5-14B-HF|134/150|True|0.18395991122350097|0.5751933643197122|0.43377314030804426|
|V8_f6750e2792f49a106b|InternVL3.5-14B-HF|140/150|True|0.18766600313596427|0.5817246849136034|0.42319315713338285|
|V8_a1177ec3c066da7288|Gemma 4 31B-it|137/150|True|0.019102160681504755|0.5289611460780851|0.46105835782595966|
|V8_16f790da645aa4e699|InternVL3.5-14B-HF|139/150|True|0.18394732038800915|0.5630529315320277|0.4213929950532272|

## commercial50: COMPLETE
- gpt-6.1-sol / FDA2: 34 hits, 50/50 completed; ASR=0.68
- gpt-6.1-sol / FDA3: 32 hits, 50/50 completed; ASR=0.64
- gpt-6.1-sol / FDA4: 36 hits, 50/50 completed; ASR=0.72
- gpt-6.1-sol / selected4: 36 hits, 50/50 completed; ASR=0.72
- gpt-6.1-sol / selected_prior: 35 hits, 50/50 completed; ASR=0.7
- gpt-6.1-sol / uniform2: 31 hits, 50/50 completed; ASR=0.62
- gpt-6.1-sol / uniform4: 35 hits, 50/50 completed; ASR=0.7

## commercial150: PARTIAL
- gpt-5.6-sol / FDA: 2 hits, 2/150 completed; ASR=None
- gpt-5.6-sol / selected: 3 hits, 3/150 completed; ASR=None
- gpt-5.6-sol / uniform: 10 hits, 12/150 completed; ASR=None
- gpt-6-sol / FDA: 123 hits, 150/150 completed; ASR=0.82
- gpt-6-sol / selected: 130 hits, 150/150 completed; ASR=0.8666666666666667
- gpt-6-sol / uniform: 127 hits, 150/150 completed; ASR=0.8466666666666667
- gpt-6.1-sol / FDA: 106 hits, 150/150 completed; ASR=0.7066666666666667
- gpt-6.1-sol / selected: 109 hits, 150/150 completed; ASR=0.7266666666666667
- gpt-6.1-sol / uniform: 105 hits, 150/150 completed; ASR=0.7

## ratio30: COMPLETE
- gpt-6.1-sol / pull_0.0: 3 hits, 30/30 completed; ASR=0.1
- gpt-6.1-sol / pull_0.25: 3 hits, 30/30 completed; ASR=0.1
- gpt-6.1-sol / pull_0.5: 19 hits, 30/30 completed; ASR=0.6333333333333333
- gpt-6.1-sol / pull_0.75: 17 hits, 30/30 completed; ASR=0.5666666666666667
- gpt-6.1-sol / pull_1.0: 17 hits, 30/30 completed; ASR=0.5666666666666667

Missing measurements remain N/A, never zero. Closed-model target DeltaR/CKA/RSA are unavailable. Correlations are exploratory and correlated across shared images/models. No causality claim.
PCA/t-SNE are fitted jointly per model/class pair; missing reverse-direction attacks are explicitly absent, not fabricated. No new reverse attack campaign is claimed.
Complete numerical receipts, gradient traces, configurations and image-level results are in the uploaded compressed data bundle; full PNGs and raw CLI traces stay on the persistent root disk.
