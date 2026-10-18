# Initial native-output validation pilot

The frozen A3 multi-layer margin + DI ensemble (CLIP/Qwen2B/InternVL2B) was evaluated on ten deterministic development images, two per direction. Held-out validation models: Qwen9B 8/10, InternVL8B 10/10, Gemma E4B 8/10. Each model classified all ten clean images correctly; none originally predicted the intended target class.

These are actual generated-label successes on development surrogates, not reference-feature class hits and not final Qwen27B/InternVL14B/Gemma31B results. N=2/direction is a pilot with substantial uncertainty; it does not establish the 15/30 formal success criterion. All floating model parameters passed BF16 and no-quantization audits. Exact per-image raw answers are in the native_validation JSON files; direction counts are in initial_native_baseline.csv.

The broad experiment queue is paused. The focused research loop tests ten-image pilots, alternative compositions and representation/optimizer variants before any further broad ablation. Final targets remain evaluation-only after method and PNG freezing.
