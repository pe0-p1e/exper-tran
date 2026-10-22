# Paired final150 interpretation

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
