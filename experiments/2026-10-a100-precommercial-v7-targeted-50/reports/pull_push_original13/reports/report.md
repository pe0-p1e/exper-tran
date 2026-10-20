# Original Multiple Proxy table: four-layer balanced Pull/Push

Fixed original13 unique workbook rows,65 directions,1950 image-evaluation rows. Same ground-truth cohorts, revisions, prompt and strict parser. New settings: uniform four-layer pure Pull/Push.5/.5, DI, equal normalized gradients,50 steps, BF16 weights. No commercial evaluation.

| Ensemble | Target | Status | New hits /150 | Old A3 hits /150 | V6 hits /150 |
|---|---|---|---:|---:|---:|
| E01 | Qwen3.5-27B | COMPLETE | 96/150 | 92/150 | 0/150 |
| E02 | Qwen3.5-27B | COMPLETE | 123/150 | 117/150 | 2/150 |
| E03 | Qwen3.5-27B | COMPLETE | 132/150 | 129/150 | 5/150 |
| E04 | Gemma 4 31B-it | COMPLETE | 74/150 | 74/150 | 0/150 |
| E05 | Gemma 4 31B-it | COMPLETE | 81/150 | 94/150 | 1/150 |
| E06 | Gemma 4 31B-it | COMPLETE | 101/150 | 108/150 | 2/150 |
| E07 | InternVL3.5-14B-HF | COMPLETE | 86/150 | 93/150 | 8/150 |
| E08 | InternVL3.5-14B-HF | COMPLETE | 90/150 | 93/150 | 13/150 |
| E09 | InternVL3.5-14B-HF | COMPLETE | 92/150 | N/A/150 | 21/150 |
| E10 | Qwen3.5-27B | COMPLETE | 116/150 | 117/150 | 4/150 |
| E11 | Qwen3.5-27B | COMPLETE | 129/150 | 131/150 | 10/150 |
| E12 | Qwen3.5-27B | COMPLETE | 132/150 | 135/150 | 20/150 |
| E13 | Qwen3.5-27B | COMPLETE | 135/150 | 138/150 | 8/150 |

Missing/partial results are not zero. Old A3 changes global loss and patch weighting relative to new PP; historical V6 additionally changes layers/DI. Neither contrast isolates the Pull/Push coefficient. Use the matched development diagnostic studies for component-specific evidence. Paired comparisons require identical150 image IDs. These final-target results do not tune the next attack or commercial proxy selection.
