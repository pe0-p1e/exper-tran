# V8 final experiment results

Updated UTC: 2026-10-22 02:44:25

Status: PARTIAL

## Protocol and selection

Fixed 1:1 cosine-prototype Pull–Push; 48 disjoint references/class; deepest and intermediate native spatial-token mean pooling; four layers per proxy in the three frozen finalists. Each target uses CLIP plus two VLM proxies outside its entire target family. Equal per-image L1-normalized proxy gradient averaging; BF16 unquantized open weights; 50 steps; 224px; epsilon16/255; alpha1/255; momentum1; DI; deterministic seed42.
The user authorized selecting selected4, FDA4 and uniform4 with seven predeclared GPT6.1-low 50-image development comparisons. This is commercial-feedback-assisted development, not untouched confirmation. Final150 has disjoint IDs/pixels and was not used for retuning. Generic commercial packets use the qwen-style CLIP/InternVL2B/GemmaE2B proxy pool.
Selected4 uses independently ranked native layers from the historical ten-image development scan; FDA4 uses conditional greedy joint-loss additions. Both are adaptations evaluated here, not faithful FDA auxiliary-classifier reproduction. All final methods use three proxies and four layers/proxy, so these final rows do not demonstrate lower compute for selected4.
Commercial target names are explicit requested Codex CLI slugs, all low effort via the existing subscription. The backend snapshot is undisclosed by this CLI; no stronger server-version claim is made. Client/version/access evidence is in commercial_preflight.json and retained raw traces.

## Final150 target success over all ground-truth-selected images

|Target|Method|Hits / planned N|Observed pairs|ASR|Wilson 95% CI|
|---|---|---:|---:|---:|---|
|Qwen3.5-27B|uniform|138/150|150|92.00%|[0.865, 0.954]|
|Qwen3.5-27B|selected|137/150|150|91.33%|[0.857, 0.949]|
|Qwen3.5-27B|FDA|137/150|150|91.33%|[0.857, 0.949]|
|InternVL3.5-14B-HF|uniform|139/150|150|92.67%|[0.873, 0.959]|
|InternVL3.5-14B-HF|selected|140/150|150|93.33%|[0.882, 0.963]|
|InternVL3.5-14B-HF|FDA|134/150|150|89.33%|[0.834, 0.933]|
|Gemma 4 31B-it|uniform|137/150|150|91.33%|[0.857, 0.949]|
|Gemma 4 31B-it|selected|134/150|150|89.33%|[0.834, 0.933]|
|Gemma 4 31B-it|FDA|139/150|150|92.67%|[0.873, 0.959]|
|gpt-5.6-sol|FDA|2/150|2|N/A|N/A|
|gpt-5.6-sol|selected|3/150|3|N/A|N/A|
|gpt-5.6-sol|uniform|8/150|10|N/A|N/A|
|gpt-6-sol|FDA|123/150|150|82.00%|[0.751, 0.873]|
|gpt-6-sol|selected|130/150|150|86.67%|[0.803, 0.912]|
|gpt-6-sol|uniform|127/150|150|84.67%|[0.780, 0.896]|
|gpt-6.1-sol|FDA|106/150|150|70.67%|[0.629, 0.774]|
|gpt-6.1-sol|selected|109/150|150|72.67%|[0.650, 0.792]|
|gpt-6.1-sol|uniform|105/150|150|70.00%|[0.622, 0.768]|

## Clean baseline and parser diagnostics

These diagnostics do not change image eligibility or the primary denominator. New targeted hits exclude images already predicted as the desired target when clean.

|Target|Method|Clean correct / observed|Clean already-target|New targeted hits|Invalid clean / adv|
|---|---|---:|---:|---:|---:|
|Qwen3.5-27B|uniform|150/150|0|138|0/0|
|Qwen3.5-27B|selected|150/150|0|137|0/0|
|Qwen3.5-27B|FDA|150/150|0|137|0/0|
|InternVL3.5-14B-HF|uniform|138/150|0|139|12/1|
|InternVL3.5-14B-HF|selected|138/150|0|140|12/0|
|InternVL3.5-14B-HF|FDA|138/150|0|134|12/1|
|Gemma 4 31B-it|uniform|150/150|0|137|0/2|
|Gemma 4 31B-it|selected|150/150|0|134|0/3|
|Gemma 4 31B-it|FDA|150/150|0|139|0/3|
|gpt-5.6-sol|FDA|2/2|0|2|0/0|
|gpt-5.6-sol|selected|3/3|0|3|0/0|
|gpt-5.6-sol|uniform|10/10|0|8|0/0|
|gpt-6-sol|FDA|149/150|0|123|0/0|
|gpt-6-sol|selected|149/150|0|130|0/0|
|gpt-6-sol|uniform|149/150|0|127|0/0|
|gpt-6.1-sol|FDA|150/150|0|106|0/0|
|gpt-6.1-sol|selected|150/150|0|109|0/0|
|gpt-6.1-sol|uniform|150/150|0|105|0/0|

## Direction breakdown

|Target|Method|Direction|Hits / planned N|Observed|
|---|---|---|---:|---:|
|Qwen3.5-27B|uniform|T01|30/30|30|
|Qwen3.5-27B|uniform|T03|29/30|30|
|Qwen3.5-27B|uniform|T05|25/30|30|
|Qwen3.5-27B|uniform|T07|30/30|30|
|Qwen3.5-27B|uniform|T09|24/30|30|
|Qwen3.5-27B|selected|T01|30/30|30|
|Qwen3.5-27B|selected|T03|28/30|30|
|Qwen3.5-27B|selected|T05|26/30|30|
|Qwen3.5-27B|selected|T07|30/30|30|
|Qwen3.5-27B|selected|T09|23/30|30|
|Qwen3.5-27B|FDA|T01|30/30|30|
|Qwen3.5-27B|FDA|T03|29/30|30|
|Qwen3.5-27B|FDA|T05|27/30|30|
|Qwen3.5-27B|FDA|T07|30/30|30|
|Qwen3.5-27B|FDA|T09|21/30|30|
|InternVL3.5-14B-HF|uniform|T01|30/30|30|
|InternVL3.5-14B-HF|uniform|T03|28/30|30|
|InternVL3.5-14B-HF|uniform|T05|28/30|30|
|InternVL3.5-14B-HF|uniform|T07|30/30|30|
|InternVL3.5-14B-HF|uniform|T09|23/30|30|
|InternVL3.5-14B-HF|selected|T01|30/30|30|
|InternVL3.5-14B-HF|selected|T03|28/30|30|
|InternVL3.5-14B-HF|selected|T05|29/30|30|
|InternVL3.5-14B-HF|selected|T07|30/30|30|
|InternVL3.5-14B-HF|selected|T09|23/30|30|
|InternVL3.5-14B-HF|FDA|T01|30/30|30|
|InternVL3.5-14B-HF|FDA|T03|27/30|30|
|InternVL3.5-14B-HF|FDA|T05|27/30|30|
|InternVL3.5-14B-HF|FDA|T07|30/30|30|
|InternVL3.5-14B-HF|FDA|T09|20/30|30|
|Gemma 4 31B-it|uniform|T01|30/30|30|
|Gemma 4 31B-it|uniform|T03|29/30|30|
|Gemma 4 31B-it|uniform|T05|24/30|30|
|Gemma 4 31B-it|uniform|T07|30/30|30|
|Gemma 4 31B-it|uniform|T09|24/30|30|
|Gemma 4 31B-it|selected|T01|29/30|30|
|Gemma 4 31B-it|selected|T03|29/30|30|
|Gemma 4 31B-it|selected|T05|21/30|30|
|Gemma 4 31B-it|selected|T07|30/30|30|
|Gemma 4 31B-it|selected|T09|25/30|30|
|Gemma 4 31B-it|FDA|T01|30/30|30|
|Gemma 4 31B-it|FDA|T03|30/30|30|
|Gemma 4 31B-it|FDA|T05|24/30|30|
|Gemma 4 31B-it|FDA|T07|30/30|30|
|Gemma 4 31B-it|FDA|T09|25/30|30|
|gpt-5.6-sol|FDA|T01|1/30|1|
|gpt-5.6-sol|FDA|T03|1/30|1|
|gpt-5.6-sol|FDA|T05|0/30|0|
|gpt-5.6-sol|FDA|T07|0/30|0|
|gpt-5.6-sol|FDA|T09|0/30|0|
|gpt-5.6-sol|selected|T01|1/30|1|
|gpt-5.6-sol|selected|T03|1/30|1|
|gpt-5.6-sol|selected|T05|1/30|1|
|gpt-5.6-sol|selected|T07|0/30|0|
|gpt-5.6-sol|selected|T09|0/30|0|
|gpt-5.6-sol|uniform|T01|2/30|2|
|gpt-5.6-sol|uniform|T03|2/30|2|
|gpt-5.6-sol|uniform|T05|2/30|2|
|gpt-5.6-sol|uniform|T07|1/30|2|
|gpt-5.6-sol|uniform|T09|1/30|2|
|gpt-6-sol|FDA|T01|29/30|30|
|gpt-6-sol|FDA|T03|26/30|30|
|gpt-6-sol|FDA|T05|21/30|30|
|gpt-6-sol|FDA|T07|30/30|30|
|gpt-6-sol|FDA|T09|17/30|30|
|gpt-6-sol|selected|T01|29/30|30|
|gpt-6-sol|selected|T03|27/30|30|
|gpt-6-sol|selected|T05|24/30|30|
|gpt-6-sol|selected|T07|30/30|30|
|gpt-6-sol|selected|T09|20/30|30|
|gpt-6-sol|uniform|T01|28/30|30|
|gpt-6-sol|uniform|T03|26/30|30|
|gpt-6-sol|uniform|T05|24/30|30|
|gpt-6-sol|uniform|T07|30/30|30|
|gpt-6-sol|uniform|T09|19/30|30|
|gpt-6.1-sol|FDA|T01|23/30|30|
|gpt-6.1-sol|FDA|T03|22/30|30|
|gpt-6.1-sol|FDA|T05|19/30|30|
|gpt-6.1-sol|FDA|T07|27/30|30|
|gpt-6.1-sol|FDA|T09|15/30|30|
|gpt-6.1-sol|selected|T01|24/30|30|
|gpt-6.1-sol|selected|T03|25/30|30|
|gpt-6.1-sol|selected|T05|21/30|30|
|gpt-6.1-sol|selected|T07|28/30|30|
|gpt-6.1-sol|selected|T09|11/30|30|
|gpt-6.1-sol|uniform|T01|23/30|30|
|gpt-6.1-sol|uniform|T03|22/30|30|
|gpt-6.1-sol|uniform|T05|20/30|30|
|gpt-6.1-sol|uniform|T07|27/30|30|
|gpt-6.1-sol|uniform|T09|13/30|30|

## Additional matched30/50 held-out surrogate validation

These development rows are method-selection evidence, not final-target confirmation.

|Validator|Method|Split|Hits / N|Observed|
|---|---|---|---:|---:|
|Qwen3.5-9B|selected4|development_explore|27/30|30|
|Qwen3.5-9B|FDA3|development_explore|26/30|30|
|Qwen3.5-9B|FDA4|development_explore|28/30|30|
|Qwen3.5-9B|selected4|development_confirm|45/50|50|
|Qwen3.5-9B|FDA3|development_confirm|45/50|50|
|Qwen3.5-9B|FDA4|development_confirm|47/50|50|
|InternVL3.5-8B-HF|selected4|development_explore|28/30|30|
|InternVL3.5-8B-HF|FDA3|development_explore|25/30|30|
|InternVL3.5-8B-HF|FDA4|development_explore|26/30|30|
|InternVL3.5-8B-HF|selected4|development_confirm|44/50|50|
|InternVL3.5-8B-HF|FDA3|development_confirm|46/50|50|
|InternVL3.5-8B-HF|FDA4|development_confirm|42/50|50|
|Gemma 4 E4B-it|selected4|development_explore|29/30|30|
|Gemma 4 E4B-it|FDA3|development_explore|28/30|30|
|Gemma 4 E4B-it|FDA4|development_explore|28/30|30|
|Gemma 4 E4B-it|selected4|development_confirm|46/50|50|
|Gemma 4 E4B-it|FDA3|development_confirm|45/50|50|
|Gemma 4 E4B-it|FDA4|development_confirm|45/50|50|

## Uniform4 Pull/Push ratio controls on matched30

|Target|Pull|Push|Hits / N|Observed|
|---|---:|---:|---:|---:|
|Qwen3.5-27B|1|0|20/30|30|
|Qwen3.5-27B|0.25|0.75|8/30|30|
|Qwen3.5-27B|0.5|0.5|27/30|30|
|Qwen3.5-27B|0.75|0.25|26/30|30|
|Qwen3.5-27B|0|1|4/30|30|
|InternVL3.5-14B-HF|1|0|18/30|30|
|InternVL3.5-14B-HF|0.25|0.75|6/30|30|
|InternVL3.5-14B-HF|0.5|0.5|27/30|30|
|InternVL3.5-14B-HF|0.75|0.25|27/30|30|
|InternVL3.5-14B-HF|0|1|2/30|30|
|Gemma 4 31B-it|1|0|23/30|30|
|Gemma 4 31B-it|0.25|0.75|8/30|30|
|Gemma 4 31B-it|0.5|0.5|25/30|30|
|Gemma 4 31B-it|0.75|0.25|26/30|30|
|Gemma 4 31B-it|0|1|2/30|30|
|gpt-6.1-sol|0|1|3/30|30|
|gpt-6.1-sol|0.25|0.75|3/30|30|
|gpt-6.1-sol|0.5|0.5|19/30|30|
|gpt-6.1-sol|0.75|0.25|17/30|30|
|gpt-6.1-sol|1|0|17/30|30|

## Statistical interpretation

Pure 1:1 Pull–Push, three legal proxies, four layers/model. GPT50 selected among predeclared candidates; final150 was evaluated after freeze.
Tests use paired image outcomes, exact two-sided McNemar, Holm within target; 95% paired-bootstrap intervals stratify the five directions. No pooling of correlated target-image rows.

|Target|Comparison A vs B|Difference pp|95% CI pp|McNemar p|Holm p|
|---|---|---:|---|---:|---:|
|Qwen3.5-27B|FDA vs selected|0.00|[-3.33, 3.33]|1|1|
|Qwen3.5-27B|FDA vs uniform|-0.67|[-4.00, 2.67]|1|1|
|Qwen3.5-27B|selected vs uniform|-0.67|[-4.00, 2.67]|1|1|
|InternVL3.5-14B-HF|FDA vs selected|-4.00|[-7.33, -1.33]|0.03125|0.09375|
|InternVL3.5-14B-HF|FDA vs uniform|-3.33|[-6.67, 0.00]|0.125|0.25|
|InternVL3.5-14B-HF|selected vs uniform|0.67|[-1.33, 2.67]|1|1|
|Gemma 4 31B-it|FDA vs selected|3.33|[0.67, 6.67]|0.0625|0.1875|
|Gemma 4 31B-it|FDA vs uniform|1.33|[-2.00, 4.67]|0.6875|1|
|Gemma 4 31B-it|selected vs uniform|-2.00|[-6.00, 2.00]|0.5078|1|
|gpt-6-sol|FDA vs selected|-4.67|[-8.67, -0.67]|0.06543|0.1963|
|gpt-6-sol|FDA vs uniform|-2.67|[-6.67, 1.33]|0.3438|0.6875|
|gpt-6-sol|selected vs uniform|2.00|[-1.33, 5.33]|0.4531|0.6875|
|gpt-6.1-sol|FDA vs selected|-2.00|[-7.33, 2.67]|0.6072|1|
|gpt-6.1-sol|FDA vs uniform|0.67|[-4.67, 6.00]|1|1|
|gpt-6.1-sol|selected vs uniform|2.67|[-2.00, 7.33]|0.424|1|

Uniform4 remains a strong baseline. Differences of a few hits do not establish superiority; non-significance does not establish equivalence.
FDA-inspired selection is an adaptation, not reproduction of auxiliary-classifier FDA. CKA/RSA are descriptive and cannot establish transfer mechanism alone.
Interpret ratios only on their matched30 cohort; final150 and historical dev cohorts have different difficulty. Missing responses are unscored and never zeros.

Integrity checks: PASS; incomplete conditions: 3.

## Measurements and reproducibility

Independently verified 2520 unique frozen attack receipts and decoded PNG pixel bounds. Recorded BF16 loads: 1955; non-pass events: 0. Full parameter requirements apply only to inspectable open weights; closed weights are unavailable.
Sum of per-image measured attack execution: 27.25 worker-hours; backwards: 378000. Concurrent worker durations are not GPU walltime, and exclude offline selection, cache creation and failed attempts.
Exact open-model revisions are in reports/model_revisions.json and all_BF16_load_audits.json.gz; each selected native layer is in final_method_lock.json. results.csv/per_image.csv/per_direction.csv contain open target DeltaR, margins and cross-model CKA/RSA. confidence_intervals.csv, paired_comparisons.csv, completion_matrix.csv and png_integrity.csv provide independent audit outputs.
projections/ exports joint PCA and t-SNE coordinates plus images for every available final target/method/direction. Each fit includes both reference classes plus clean/adversarial images; reverse attacks are absent. t-SNE distances are qualitative. CKA/RSA are means across proxy-layer/target reference geometry; individual values are exported. Closed-model DeltaR/CKA/RSA are N/A.
examples/paired_examples.png displays the first frozen image per direction, with clean/Uniform/Selected/FDA images side by side; no success-based example selection. Exact image IDs and PNG hashes are in examples/index.json.
Full PNGs, raw response traces, failed attempts and restartable checkpoints remain on the persistent root disk under /home/ubuntu/exper-tran/outputs/a100_precommercial_v8_joint_selection24h. GitHub carries code and compact complete numerical data, not the full image corpus.

## Limits and outstanding work

Only a fixed five-direction ten-class task and one primary seed were evaluated here. These results do not establish open-ended VQA, category generality, universal transfer or a causal mechanism. Small method differences are not automatically significant; shared images across models must not be treated as independent extra samples.
Incomplete planned conditions: 3. Missing results are N/A; no failed or incomplete response is counted as a complete150 run.
