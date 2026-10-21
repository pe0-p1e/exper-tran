# Post-layer proxy selection: prepared, not yet dispatched

User2026-10-21: prepare proxy selection from the reviewed literature, run after layer selection, and make large-model evaluation the meaningful result. This directory is isolated from the active scientific source lock. Preparation performs no inference, GPU admission, model downloads, authentication or commercial calls. The active six-worker campaign continues.

## Order and model roles

1. Finish layer-method development. Freeze per-model native layers with exact revisions and completed evidence IDs. Never change layers while comparing proxy identities; do not fill missing taps by guessing quartiles.
2. Use small development validators only for inexpensive screening/conditional search. Their near100% scores are not the main outcome.
3. Freeze exactly one selected method and a matched control before big-model responses. Freeze confirmation PNGs. Evaluate **Qwen3.5-27B,InternVL3.5-14B-HF,Gemma4-31B-it**, whose exact locked snapshots are already cached. No size/precision fallback and no feedback from these responses for proxy selection.
4. Begin with10 paired images (two/direction), then50 (ten/direction); the10 are part of the50, not additional independent cases. At most two methods gives300 model–image pairs on the full panel,450 actual clean/adv requests with exact clean reuse. Keep the150-image formal Stage3 set sealed. These large checkpoints have historical experiment exposure; do not claim unseen models.
5. Full component ablations remain Stage3. No commercial calls are part of this preparation.

## Controlled core matrix

V=CLIP-L/14. Every checkpoint uses the same frozen model-specific layer policy across all rows,50-step1:1 PP,DI,BF16,equal normalized fusion and fixed GT cohorts.

| Large confirmation target | Development screen | A | B |
|---|---|---|---|
|Qwen27B|Qwen9B|InternVL2B|Gemma E2B|
|InternVL14B|InternVL8B|Qwen2B|Gemma E2B|
|Gemma31B|Gemma E4B|Qwen2B|InternVL2B|

For each family compile V,A,B,V+A,V+B,A+B,V+A+B:21 candidate conditions, **not21 new runs**. Reuse attacks only after checking science, cohort, exact taps, checkpoint/source locks and PNG bounds/hash receipts. Retain the prior model order and identity. Reuse evaluations only when the actual validator/revision/protocol also match. A small-model response never substitutes for a large-model result.

The core matrix is evidence for bounded method selection, not a new full factorial ablation. Preregister no more than four newly generated static conditions/family before reassessing throughput; defer surplus rows if necessary with explicit status. Existing uniform/matched singles are the baseline. Conditional selection evaluates real joint attacks; source effectiveness/CKA/conflict only screen candidates. Compare fixedK=2 identity changes before increasingK. Preserve gained AND lost image counts.

## Literature-informed selector

- **FDA-inspired conditional utility:** with layers fixed, treat an entire proxy as the add/remove unit. Choose additions from real held-out development joint attack gains, not singleton ranking or union of successful images.
- **AdaEA-inspired source contribution diagnostics:** on identical x/DI state, measure cross-objective gains from each proxy update. Primary fusion remains equal normalized gradients; no claim to reproduce AdaEA output fusion.
- **CWA-informed compatibility:** measure common-descent and negative-dot fractions. Low CKA/orthogonal gradients alone are not useful complementarity. Flatness or SAM is not claimed without separate measurements.
- **ICML2025 ensemble theory:** strength/diversity/complexity motivates measured performance–compute Pareto comparisons, not a theorem that our CKA metric is optimal.
- **FiT:** original uses preliminary target queries. Only development-validation analogy is lawful here; final big-model responses cannot select proxies.
- **SEA:** prepare a separate future temporal prototype-gradient adaptation (e.g. active2 from pool4, matched gradient-call budget). It requires an explicit worker adapter and tests. The current compiler does not label static subsets or arbitrary round-robin as SEA, and does not dispatch this optional arm.

Sources and distinctions: ../PROXY_SELECTION_LITERATURE.md . No proxy-paper reproduction result is claimed by this preparation.

## Expanded pool, bounded active set

`candidate_catalog.json` records pinned eligible CLIP-B/32,B/16,L/14,L/14@336,DINOv2-Base/Large,SigLIP2 and small/medium cross-family VLMs. Whole target-family exclusion applies even to disabled candidates. Extensions remain disabled until each new model has an audited frozen layer policy and cache identity. No Cartesian enumeration. At most6 vision encoders (up to4CLIP,2DINO;SigLIP replaces a slot),2VLMs,total8; initial method search remains much smaller. These are upper limits, not a request to load8models concurrently.

## Scientific cache contract

A model's selection gradient must be **the gradient of its mean layer losses, then per-image L1 normalization**. Averaging already normalized single-layer gradients is not equivalent and cannot reuse that cache. CKA uses aligned480-reference layer-Gram aggregation with fixed normalization; no raw cross-dimensional cosine. Directed gains use only surrogate PP objectives. The candidate, cohort, reference, layer, precision and code identities must match before cache reuse.

## Concrete preparation tools and remaining gate

The CPU-only entry is `run_proxy_selection_followup.sh`; default prepares inventory/templates, `--status` reads preparation status, `--layer-lock PATH` validates a completed per-model layer policy and materializes existing-scheduler task declarations. `prepare.py` also provides immutable `freeze_methods` and `frozen_large_packet` helpers; large target evaluation packets bind every existing PNG and the frozen method hash, with a different evaluation model rather than regenerated attacks.

This is a compiler/handoff package, not a second campaign owner. It is **not enqueued** while the layer freeze is missing; the existing admission owner must dispatch its attack/evaluation tasks after the gate. The preparation does not silently insert a new stage into today's frozen master pipeline or claim an untested large-target evaluator has already run. No user intermediate command is requested; the connected agent handles the handoff when the layer result is ready.

Large-model evaluation must group targets to minimize reloads, finish/checkpoint owned attack workers before heavy loading, audit all actual BF16/no-quant parameters and calibrate generation batches. Up to6 workers share the current owner; A100 headroom and≤72GiB aggregate reservations override worker count. Never evict foreign work or use reduced precision to fit a target.

Large models can still saturate on an easy fixed ten-class task. Do not choose hard images post hoc from target errors. If saturation persists, predeclare broader classes/COCO tasks for Stage3 and report uncertainty honestly.
