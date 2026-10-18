# Follow-up benchmarks (deferred until core V7 is validated)

Priority confirmed 2026-10-18: finish the original ImageNet V7 representation-objective and multi-proxy study first. These extensions are planned, not executed results.

1. MS COCO: ground-truth-only, deterministic disjoint reference/development/test manifests; multi-label-aware source/target selection, target-absence audit, single proxy, embedding layers, multiple proxy and loss/fusion ablations. Freeze categories, task prompts, sample counts and metrics before evaluation. Never relabel an ImageNet experiment as COCO.
2. LLaVA-Bench (COCO): acquire original questions, images and reference answers with immutable provenance. Distinguish it from LLaVA-Bench In-the-Wild. Preserve official free-form questions. Report clean/attacked responses and task degradation; an open-weight judge is a separate score, not the official GPT-4 judge result.
3. Web scenarios: open-weight product identification, image-driven search/retrieval and web-image question answering. Use controlled local pages/catalogs, frozen relevance labels, targeted wrong-item retrieval/answer rates and clean utility. Test JPEG compression and browser resize as separately labeled delivery transformations. Treat representation attack components as hypotheses; document novelty against original literature.
4. Commercial models: user will manually upload frozen clean/adversarial images. Prepare a reproducible pack with fixed prompts, paired images, image hashes, model/version/time fields, parser rubric and raw-response recording. No commercial API calls or fabricated manual measurements.

All follow-up work keeps V6 immutable and uses separate outputs. The core V7 primary budget, 150 locked ImageNet images and final target holdout remain unchanged.

Sources: https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_Bench.md ; https://cocodataset.org/
