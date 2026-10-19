# Multi-proxy selection ablation

V/A/B use target-specific family-disjoint proxy pools in LATEST_EXPERIMENT_FRAMEWORK.md. Loss, DI, layers and attack budget are fixed. Pending/partial cells are not zero.

| Target | Subset | K | Hits/N | ASR | Status |
|---|---|---|---|---|---|
| Qwen3.5-27B | V | 1 | 0/0 | pending | pending |
| Qwen3.5-27B | A | 1 | 0/0 | pending | pending |
| Qwen3.5-27B | B | 1 | 0/0 | pending | pending |
| Qwen3.5-27B | V+A | 2 | 0/0 | pending | pending |
| Qwen3.5-27B | V+B | 2 | 0/0 | pending | pending |
| Qwen3.5-27B | A+B | 2 | 0/0 | pending | pending |
| Qwen3.5-27B | V+A+B | 3 | 0/0 | pending | pending |
| InternVL3.5-14B-HF | V | 1 | 0/0 | pending | pending |
| InternVL3.5-14B-HF | A | 1 | 0/0 | pending | pending |
| InternVL3.5-14B-HF | B | 1 | 0/0 | pending | pending |
| InternVL3.5-14B-HF | V+A | 2 | 0/0 | pending | pending |
| InternVL3.5-14B-HF | V+B | 2 | 0/0 | pending | pending |
| InternVL3.5-14B-HF | A+B | 2 | 0/0 | pending | pending |
| InternVL3.5-14B-HF | V+A+B | 3 | 0/0 | pending | pending |
| Gemma 4 31B-it | V | 1 | 0/0 | pending | pending |
| Gemma 4 31B-it | A | 1 | 74/150 | 49.3% | complete |
| Gemma 4 31B-it | B | 1 | 0/0 | pending | pending |
| Gemma 4 31B-it | V+A | 2 | 0/0 | pending | pending |
| Gemma 4 31B-it | V+B | 2 | 0/0 | pending | pending |
| Gemma 4 31B-it | A+B | 2 | 0/0 | pending | pending |
| Gemma 4 31B-it | V+A+B | 3 | 130/150 | 86.7% | complete |

Gradient metrics may cover only completed attack batches; diagnostic_images records that scope. Negative pairwise cosine fraction is not pixelwise sign disagreement. CKA/RSA are pending until matching clean embeddings are measured. Reused historical results retain their source fingerprint. Raw scores can describe proxy selection but cannot tune another attack on the same evaluated target.
