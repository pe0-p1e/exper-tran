# Latest experiment framework and execution order

Authoritative planning update: 2026-10-19. This consolidates the user's latest
priorities. It does not change the immutable running family-disjoint campaign.
Machine-readable companion: `latest_experiment_framework.yaml`.

Latest user correction: paper drafting is paused. Work now concerns experimental
planning/code and proxy-selection tables; the research order below is unchanged.

## Execution order

1. **Finish and report the current strict cross-family correction.** Qwen27B
   excludes all Qwen attack proxies; InternVL14B excludes all InternVL proxies.
   Preserve the valid earlier Gemma31B cross-family measurements. The current
   queue has32 conditions,160 cells and4,800 image evaluations. A condition is
   not complete until all five directions have30 evaluated images and audits.
2. **Multi-proxy ablation and proxy selection.** Build the table first; complete
   missing pairwise subsets while reusing compatible completed measurements.
   Fix the loss/optimizer to isolate proxy identities and composition. Report
   whether the ensemble improves over its own strongest individual member,
   what extra proxies cost and whether redundancy/conflict explains failures.
3. **Loss/category generalization.** Freeze the method before any held-out-class
   target evaluation. Test disjoint new categories and separately expand a
   nested category dictionary10→20→50. New reference banks are allowed; retuning
   on those target outputs is not. Preparation code may be written now, but
   scientific GPU execution follows the selection study.
4. **Commercial targets.** Only after the preceding studies and ablation table
   are fixed, start the THREE-image CLI smoke, then expand. Codex/OpenAI,
   Gemini and Claude are one exact model each, with client/route/version logged.
   Antigravity CLI is the preferred Gemini/Claude initial route; AnyRouter
   Claude is a later option requiring its actual endpoint/model/access/budget.
   No commercial model has been evaluated yet. No open-ended VQA is launched.

The three-image blinded pack and CLI installation are preparation, not commercial
results. They do not authorize skipping the revised priority order. All image
selection remains GT-only; no successful-example filtering.

## Fixed core method and evaluation

Four normalized visual depths25/50/75/100%, spatial-token mean pooling, image-only
ten-class prototype margin, local patch weight0.35, DI, equal averaging of
per-image L1-normalized proxy gradients. Epsilon16/255,50 steps,alpha1/255,
momentum1, random start, seed42 and224px attack canvas. No language head or CLS
loss. Target predictions/embeddings/gradients never enter attack generation.

Every floating open-weight model parameter is BF16, with quantization forbidden.
Commercial target weight precision and architecture are unknown and cannot be
audited. Keep48 reference images/class, disjoint from the30 attacked images per
direction. Primary metric is targeted hits/N over the full frozen GT cohort;
legacy proxy-success-conditioned TASR is separately labeled and undefined
events remain N/A. Report parser failures, exact counts and uncertainty.

The five directed tasks remain goldfish→monarch butterfly,
pineapple→acoustic guitar, laptop→espresso, volcano→rocking chair and
soccer ball→school bus. They are image-class transitions, not model pairs.

## Primary multi-proxy selection table

Use three slots per target:

| Target | V | A | B |
|---|---|---|---|
| Qwen27B | CLIP ViT-L/14 | InternVL2B | Gemma E2B |
| InternVL14B | CLIP ViT-L/14 | Qwen2B | Gemma E2B |
| Gemma31B | CLIP ViT-L/14 | Qwen2B | InternVL2B |

All seven nonempty subsets V,A,B,V+A,V+B,A+B,V+A+B are distinct conditions.
This yields21 target-specific conditions,105 directed cells and3,150 image
evaluations, with compatible results reused rather than regenerated. The
running corrected study already covers V,A,B,V+A+B for Qwen and InternVL.
The historical valid Gemma full ensemble and matched Qwen single may be reused
after settings/cohort/PNG/prompt provenance checks. Reused implementation
fingerprints and compute measurements must remain visible.

Required table columns:

- Exact proxy names andK, target and source→target class.
- Targeted hits/N, ASR, clean accuracy and clean-correct ASR.
- Gain against the best measured single **within that subset**, plus a separate
  best-in-pool comparator. Paired image-level intervals require identical IDs.
- Pairwise clean-feature CKA/RSA, pairwise gradient cosine, negative cosine
  fraction and equal-mean gradient cancellation along the attack trajectory.
- Runtime, peak VRAM and real per-model forward/backward counts.
- Status, reuse provenance, missing measurements and failures.

The seven-subset table studies the available fixed pool; it does not establish
that this pool is globally optimal. Broader replacements by SigLIP2/DINO or
different VLM scales are a separately predeclared extension, not mandatory
Cartesian expansion. A low CKA or a negative gradient cosine is not assumed to
improve transfer; that relationship must be tested.

**Selection versus description:** target-wise best observed rows can be described
after evaluation, but cannot be fed back into another attack on the same target
and called held-out performance. A deployable selector must use separate
surrogate-dev measurements, with its rule frozen before final target outputs.
For stronger unseen-family claims, exclude the target family from the tuning
validators too, and report this separately from attack-proxy family exclusion.

## Generalization and commercial tables

G1: original ten classes versus disjoint new ten, frozen method and proxy rule.
G2: nested10/20/50 options on identical held-out images/source-target pairs.
New class references instantiate the loss; this is reference-conditioned
category generalization, not reference-free zero-shot or proof of universal
open-ended task success. See `HELD_OUT_CLASS_GENERALIZATION.md`.

C0: three total images for connectivity/image-input/isolated-session checks,
clean and full attack in separate requests. Then preregister a compact component
ablation table using open-weight evidence. Commercial outcomes may explain the
locked method, not choose the winning algorithm on its final test set.
See `COMMERCIAL_PILOT.md`. Model names/aliases, dates, raw responses, request
hashes and route failures must be recorded. Any router model fallback invalidates
a fixed-model comparison unless separately reported.

## Paper evidence and delivery

The scientific narrative is: representation objective → proxy complementarity
and selection → category generalization → closed-weight transfer. Web relevance
requires a real image-mediated Web task or a carefully bounded discussion;
classification alone must not be relabeled an evaluated Web deployment.

Maintain an evidence ledger distinguishing completed, running, prepared and
planned items. Report the family-disjoint correction first when it finishes.
Save complete outputs/checkpoints on persistent root disk. Push implementation,
compact scientific data, reports and a few actual examples; omit full image
corpora, embedding caches, tokens and model weights. V6 remains immutable.
