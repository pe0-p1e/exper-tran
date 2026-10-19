# Commercial targets: three-image connectivity pilot, then frozen ablations

User override, 2026-10-19: commercial preparation is now authorized alongside
the running open-weight family-disjoint campaign. Start with THREE images total,
then expand. Antigravity CLI is the preferred initial Gemini/Claude interface;
Codex is the OpenAI candidate; AnyRouter's latest accessible Claude is a later
route. Do not run paid APIs until the actual endpoint, model, credentials and
spend limit are configured. No open-ended VQA in this phase.

## Stage 0: three images, clean versus full attack only

Use one deterministic GT-selected development image from each of T03
(pineapple to acoustic guitar), T05 (laptop to espresso), T09 (soccer ball to
school bus). Selection uses the first frozen development-manifest ID, without
filtering on any clean or attacked model response. This is a deliberately
limited difficult-direction smoke, not coverage of all five directions.

Reuse the previously surrogate-selected full-method attack on those exact
images: CLIP + Qwen2B + InternVL2B, four-layer margin, patch weight0.35, DI,
equal L1-gradient fusion, epsilon16/255, 50 steps, seed42. This smoke does not
choose an algorithm. No additional GPU attack is needed for these cached PNGs.

Three images x(clean + adversarial) x3 models =18 model-image evaluations.
Clean and adversarial images are separate fresh requests, never side-by-side
and never in the same conversation. Use opaque filenames, one image/request,
the original fixed ten-class prompt and strict original output parser.
Do not show the source/target label, experiment directory, repository, score
sheet, attack status or earlier responses to a target agent. Disable unrelated
tools, memories, search, skills and context; retain machine-readable traces.
Only direct image input or an audited read of the supplied image is permitted.

Codex/Antigravity are clients, not unique model identifiers. Record actual
model ID/version, client version, route, timestamp, prompt and PNG hashes,
reasoning/output settings and exact raw response. Do not mix CLI and API routes
as if they were one measured target. If the client cannot verify image input,
fresh context or requested-model identity, record an interface blocker rather
than a valid benchmark result. A router must not silently switch models.

Closed-weight parameter precision and architecture are unavailable: BF16 audits
apply to open-weight attack proxies only; never claim to have audited a
commercial target's weights or proved that its vision backbone is disjoint.

## Stage 1: preregister the small ablation table

Wait for the corrected open-weight campaign to decide which existing proxy
composition/method will be frozen for commercial evaluation. Use open-weight
results to formulate claims; do not preselect desired numerical conclusions.
The following is a draft seven-row table, frozen before commercial ablations:

| Row | Method | Question |
|---|---|---|
| C0 | Original deepest Pull/Push ensemble, no DI | Historical algorithm baseline; changes multiple components |
| C1 | Strongest open-validation single proxy with full method | Does the fixed ensemble improve over a strong single? |
| C2 | Full method, replace GLOBAL ten-class margin with Pull/Push | Effect of global all-class competition, preserving layers/local/DI |
| C3 | Full method, deepest layer only | Effect of multi-depth optimization |
| C4 | Full method, no patch term | Effect of local matching |
| C5 | Full method, no DI | Effect of input diversity |
| C6 | Full method | Reference method |

All rows share images, references, epsilon, steps, seed and prompts; report model
forward/backward cost as well as attack steps. Global and local margin terms
must not be conflated in C2. This draft is a future study; the open-weight
running matrix does not currently contain every one of these exact rows.

If the later cohort is15 TOTAL images, distribute3 across each of all five
directions and use development images disjoint from the final150-image cohort.
Seven rows x15 x3 =315 adversarial evaluations, plus45 reusable clean evaluations
=360 model-image evaluations before retries or controls. The initial3-image
smoke is not itself a completed ablation or an ASR claim with useful precision.

## Stage 2: final commercial evaluation

Choose/freeze the final composition using open-weight surrogate validation,
BEFORE examining commercial ablation outcomes. Commercial ablations explain the
locked method and do not start another target-feedback search. If commercial
feedback is ever used for selection, label that explicitly as query-informed
development and obtain a fresh independent final cohort; do not call it the
original zero-query transfer protocol.

Use one exact accessible vision-capable model per OpenAI/Gemini/Claude target,
record the date/model snapshot instead of an unstable 'latest' label, and
evaluate the locked method on the original five x30 final images when budget
permits. Report targeted ASR with exact numerator/denominator and per-direction
breakdowns. Historical proxy-success-conditioned TASR is a separate secondary
metric and remains N/A if its event has not been measured. Refusals/invalid
answers are not target successes; transport errors are tracked separately,
never silently dropped. Never pick images or model versions by achieved ASR.

## Current source documentation

- OpenAI image input: https://developers.openai.com/api/docs/guides/images-vision
- Codex model documentation: https://developers.openai.com/api/docs/models/gpt-5.3-codex
  (documents a specific model, not a recommendation that it is the newest).
- Antigravity CLI installation: https://antigravity.google/docs/install?tab=cli
- Antigravity headless mode: https://www.antigravity.google/docs/cli/headless/
- Gemini model versioning: https://ai.google.dev/gemini-api/docs/models
- Claude IDs: https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions
- AnyRouter routing documentation (actual user endpoint must be confirmed):
  https://docs.anyrouter.dev/guides/provider-routing
