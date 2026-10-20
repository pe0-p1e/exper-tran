# Strict cross-family V7 correction

Task: fixed ten-class targeted image recognition. No open-ended VQA.

Completed cells: 40/160. Image evaluations: 1200/4800. Recorded failed attempts: 0.

Qwen27B excludes every Qwen proxy; InternVL14B excludes every InternVL proxy. Gemma31B prior cross-family primary results are preserved. Old Qwen/InternVL measurements are family-overlap experiments, not valid strict-cross-family replacements.

The algorithm, proxy substitutions, all ablations and seeds were predeclared before corrected target evaluation. This is a protocol correction after earlier target results were observed; it is not an untouched new benchmark. No further target-feedback optimization is permitted. All ground-truth cohorts, references, prompts, epsilon16/255 and 50 steps are unchanged.

## Primary results

| Target | Direction | Hits/N | ASR |
|---|---|---|---|
| Qwen3.5-27B | T01 | 30/30 | 100.0% |
| Qwen3.5-27B | T03 | 26/30 | 86.7% |
| Qwen3.5-27B | T05 | 25/30 | 83.3% |
| Qwen3.5-27B | T07 | 30/30 | 100.0% |
| Qwen3.5-27B | T09 | 20/30 | 66.7% |
| InternVL3.5-14B-HF | T01 | 30/30 | 100.0% |
| InternVL3.5-14B-HF | T03 | 27/30 | 90.0% |
| InternVL3.5-14B-HF | T05 | 27/30 | 90.0% |
| InternVL3.5-14B-HF | T07 | 30/30 | 100.0% |
| InternVL3.5-14B-HF | T09 | 22/30 | 73.3% |
| Gemma 4 31B-it | T01 | 29/30 | 96.7% |
| Gemma 4 31B-it | T03 | 27/30 | 90.0% |
| Gemma 4 31B-it | T05 | 23/30 | 76.7% |
| Gemma 4 31B-it | T07 | 30/30 | 100.0% |
| Gemma 4 31B-it | T09 | 21/30 | 70.0% |

All 15 strict-cross-family primary cells >=15/30: True.

All matched singles, five individual layer depths, component ablations and primary seeds43/44 are reported in results/*.csv. No favorable seed replaces seed42. Original Pull/Push uses deepest layer without DI and is an historical algorithm control, not a single-component ablation. The no-patch and no-DI conditions provide matched component controls.

Undefined conditional TASR is N/A. Missing results are not zero. Geometry is restricted to the two corrected primary ensembles. No reverse-direction cohort is present, so directional asymmetry is not estimated.
