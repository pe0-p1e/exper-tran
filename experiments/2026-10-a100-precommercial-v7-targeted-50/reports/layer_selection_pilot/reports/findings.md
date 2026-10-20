# Completed layer/model pilot: independently checked findings

All10 conditions complete;100 model-image evaluation rows on10 unique historical development images. Held-out validator: Qwen3.5-9B, not final Qwen27B. Attack proxies exclude Qwen family. Fixed balanced Pull/Push, DI,50 steps, BF16 and equal independently L1-normalized proxy fusion. No final target/commercial calls.26 recorded BF16 load audits passed. Frozen-cohort identities, attack configurations, saved PNG hashes and recorded PNG bounds were independently rechecked for all10 conditions; evaluation hits were recomputed from image-level rows.

## Layer comparisons (same InternVL2B+GemmaE2B+CLIP model order)

| Layers | Hits/N |
|---|---|
|37.5%|1/10|
|62.5%|5/10|
|75%|7/10|
|87.5%|7/10|
|62.5%+87.5%|7/10|
|25%+50%+75%+100%|9/10|

75%,87.5% and the tested two-layer subset have identical per-image success masks. Adding62.5% to87.5% contributes no new successes in this pilot. The four-layer baseline gains two successes and loses none relative to75%: one additional T01 image and one T05 image. Per-direction results for the four-layer baseline are2/2,2/2,1/2,2/2,2/2; the single75% counts are1/2,2/2,0/2,2/2,2/2.

## Model comparisons (all use four uniform layers)

| Proxies | Hits/N |
|---|---|
|InternVL2B|7/10|
|GemmaE2B|4/10|
|CLIP-L/14|7/10|
|InternVL2B+GemmaE2B|7/10|
|InternVL2B+GemmaE2B+CLIP|9/10|

InternVL, CLIP and InternVL+Gemma have identical success masks; Gemma's successes are a subset. Thus even the union of separately successful InternVL and CLIP attacks is7/10, while the actual three-model jointly optimized attack achieves9/10. This is descriptive evidence worth investigating: a joint attack may succeed where neither of these individual attacks does. It does not isolate which pair is sufficient; matched InternVL+CLIP and Gemma+CLIP attacks are still missing. The unchanged InternVL+Gemma success mask does not establish their feature or gradient equivalence.

## Limits and next comparison

This is an exploratory reused development cohort with only two images/direction and one held-out surrogate. Two additional successes are not population-level evidence of a reliable20-percentage-point gain. Single-layer conditions here still contain THREE proxy models. The current paired comparisons do not prove that four layers are optimal, nor that individually strongest layers form the best subset. Next compare complementary and ranked same-cardinality pairs, including missing model pairs; confirm independently before claiming a general rule.

Campaign wallclock: approximately11min14sec (16:41:39–16:52:53 UTC), including parallel work and cached artifacts. Existing compatible attacks and the four-layer validation were reused. Reported per-condition attack seconds come from their original state receipts under different concurrency and cache conditions; they cannot be used as a controlled compute-efficiency ranking. A dedicated matched performance measurement is required.

Machine-readable evidence: conditions.csv, per_image.csv, direction_results.csv, paired_subset_results.csv, integrity_checks.csv, independent_validation.json, protocol_lock.json, model dtype audit and validation JSONs. Full PNGs stay on the persistent local disk.
