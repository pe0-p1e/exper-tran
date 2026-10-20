# Does balanced Pull/Push generalize across proxy combinations?

Same frozen ten-image historical surrogate development cohort and three held-out checkpoints. Four layers, DI, no patch, BF16,50 steps. Ratio selection never reads commercial/final-target outputs.

| Composition | Pull | Status | Target hits /30 model-image observations |
|---|---:|---|---|
| S_CLIP | 0.5 | PARTIAL | N/A/30 |
| S_CLIP | 0.75 | PARTIAL | N/A/30 |
| S_Qwen | 0.5 | PARTIAL | N/A/30 |
| S_Qwen | 0.75 | PARTIAL | N/A/30 |
| S_InternVL | 0.5 | QUEUED | N/A/30 |
| S_InternVL | 0.75 | QUEUED | N/A/30 |
| S_Gemma | 0.5 | QUEUED | N/A/30 |
| S_Gemma | 0.75 | QUEUED | N/A/30 |
| S_DINO | 0.5 | QUEUED | N/A/30 |
| S_DINO | 0.75 | QUEUED | N/A/30 |
| Qwen_InternVL | 0.5 | QUEUED | N/A/30 |
| Qwen_InternVL | 0.75 | QUEUED | N/A/30 |
| CLIP_Qwen_InternVL | 0.5 | COMPLETE | 29/30 |
| CLIP_Qwen_InternVL | 0.75 | COMPLETE | 24/30 |
| CLIP_Qwen_Gemma | 0.5 | QUEUED | N/A/30 |
| CLIP_Qwen_Gemma | 0.75 | COMPLETE | 24/30 |
| CLIP_InternVL_Gemma | 0.5 | QUEUED | N/A/30 |
| CLIP_InternVL_Gemma | 0.75 | QUEUED | N/A/30 |
| CLIP_DINO_Qwen | 0.5 | QUEUED | N/A/30 |
| CLIP_DINO_Qwen | 0.75 | QUEUED | N/A/30 |
| CLIP_DINO_InternVL | 0.5 | QUEUED | N/A/30 |
| CLIP_DINO_InternVL | 0.75 | QUEUED | N/A/30 |
| CLIP_DINO_Gemma | 0.5 | QUEUED | N/A/30 |
| CLIP_DINO_Gemma | 0.75 | QUEUED | N/A/30 |
| CLIP_DINO_Qwen_InternVL | 0.5 | QUEUED | N/A/30 |
| CLIP_DINO_Qwen_InternVL | 0.75 | QUEUED | N/A/30 |

Compare ratios within each composition using paired_weights.csv, and model/direction details in directions.csv. Observations from three validators on the same image are dependent; bootstrap resamples ten image clusters. Partial results are not full-cohort scores. Matched singles supply composition controls; no target-side method selection or commercial calls.
