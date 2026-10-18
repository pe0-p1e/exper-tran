# Execution priority update, 2026-10-18

User requested algorithm/composition optimization first, initially only 8–10 images, then small ablations after success. The earlier 31-condition full-test queue is paused; all its completed artifacts and original method lock remain unchanged. No final target evaluation from that queue has occurred.

Default `run_all_v7.sh` now delegates to `run_focus_v7.sh`. `--legacy-full` is retained for explicitly restoring the old broad queue, not launched automatically.

## Small-pilot protocol

- Use the first two deterministic development images/direction: five directions × two = ten images per validation model. Never select on any prediction.
- Keep 48 disjoint references/class, strict BF16, 224px canvas, 16/255, 50 steps, alpha=1/255 and seed=42.
- Score actual generated labels on held-out Qwen9B, InternVL8B and Gemma E4B. Exclude these validators from proxy combinations. These are development models, not final Qwen27B/InternVL14B/Gemma31B.
- Record unconditional hits/denominator. The old 93% reference-feature hit rate is only a representation diagnostic, not native-output ASR and not legacy conditional TASR.
- The ten-image pilot cannot establish the formal >=15/30 performance target. Expand confirmation only after a promising pilot; do not replace the locked 30-image primary benchmark.

## Focused search

1. Existing four-layer margin+DI baseline.
2. Reference-only feature centering/variance scaling; temperatures and hard negatives; distributional reference affinities; layer weights; actual visual-projector image tokens.
3. Alternative 2/3/4-family/vision-encoder combinations: CLIP, SigLIP2, DINOv2, Qwen2/4B, InternVL2/4B, Gemma E2B. No final target participates.
4. Nesterov lookahead, spectral EOT, variance-tuned gradients and adaptive/conflict fusion, chosen by independent native surrogate performance. Additional forwards/backwards are recorded and extra compute is labeled.
5. Freeze the best algorithm before a single ten-image final-target pilot: first two of the immutable test images/direction. All PNGs freeze before any final target is loaded. No target feedback may change algorithms, weights, cohorts or PNGs. Resume after target access only completes the frozen evaluation; it must not regenerate images or restart adaptive search.
6. Two small prespecified ablations only after native pilot gates pass; otherwise record the actual shortfall. Broad matrix remains deferred.

## Literature and scope

- ILA (ICCV 2019): intermediate feature directions can help transfer. This does not prove that our layer combinations work. https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Enhancing_Adversarial_Example_Transferability_With_an_Intermediate_Level_Attack_ICCV_2019_paper.pdf
- IPGA (2025): motivates studying projector image tokens. Our Qwen/InternVL projector extraction uses their real merger/projector, never assumes a Q-Former, and is not a reproduction of IPGA. https://arxiv.org/abs/2508.13739
- SSA (ECCV 2022): motivates frequency-space diversity. Our FFT masking is an explicitly different adaptation from its DCT scheme; extra EOT compute is counted. https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/974_ECCV_2022_paper.php
- Variance Tuning (CVPR 2021): previous-step neighborhood gradient correction. Our projected neighborhood variant applies surrogate visual-representation losses. https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.html
- AdaEA/PCGrad/DI remain inspired adaptations, not exact classifier-task reproductions. Their literature percentages are not comparable to the current targeted MLLM benchmark.

Centering and variance scaling are reference-fitted numerical/geometry hypotheses, not an asserted novel method or proven causal explanation. Every unsuccessful candidate is preserved.
