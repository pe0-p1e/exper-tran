# Unified commercial evaluation, surrogate controls and mechanism analysis

The current user authorized parallel evaluation and research on2026-10-19.
One local GPU controller owns the72GiB reservation ledger and admits at most
FOUR total GPU workers, including the original family-disjoint campaign.
The original campaign is unchanged. The new controller adopts supplemental
children without stopping their attack batches. The existing18-hour deadline
is preserved, not reset. Incomplete conditions remain resumable.

Entry point: `bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_unified_study.sh`.
`--preflight` verifies locks/cohorts without model inference.

## Independent commercial lane

All versions consume the SAME frozen PNGs. No regeneration depends on the
commercial target. First evaluate ten images (two per direction), publish
the report, then automatically evaluate150 fixed test images for every model
with20 verified ten-image requests. This gate uses interface completeness,
never ASR. A low-scoring valid model is retained. Unavailable interfaces are
reported separately. Negative predictions are never retried.

`commercial_validation_campaign.py` runs this chain. The lower-level
`run_commercial_matrix.sh` supports `--stage smoke3|dev10|test150` and
`--preflight`; users need not launch stages separately.

The verified client catalogs currently expose12 enabled conditions:

| Route | Requested model ID | Requested reasoning |
|---|---|---|
| Codex CLI0.163.0 | gpt-6.1-sol | low |
| Codex CLI0.163.0 | gpt-6-sol | low |
| Codex CLI0.163.0 | gpt-5.6-sol | low |
| Codex CLI0.163.0 | gpt-6-astra | low |
| Codex CLI0.163.0 | gpt-6-luna | low |
| Codex CLI0.163.0 | gpt-5.6-terra | low |
| Codex CLI0.163.0 | gpt-5.6-luna | low |
| Antigravity CLI1.4.2 | gemini-3.8-flash-low | low |
| Antigravity CLI1.4.2 | gemini-3.8-flash-medium | medium |
| Antigravity CLI1.4.2 | gemini-3.8-flash-high | high |
| Antigravity CLI1.4.2 | claude-sonnet-4-6 | provider default |
| Antigravity CLI1.4.2 | claude-opus-4-6-thinking | provider default |

Catalog visibility does not guarantee inference availability; each row still
requires actual image inference. Gemini entries are three reasoning profiles
of ONE generation, not three historical model generations. Claude's current
route rejects an explicit low effort flag before inference, so it is omitted
and the default is disclosed. Backend dated snapshots and closed model weights
are not exposed. Do not claim a closed-model BF16 audit or guaranteed family
disjointness.

All12 conditions imply240 ten-image model/image observations and3600 full-test
observations, before reuse. Model availability and actual speed determine
completion; no unmeasured walltime guarantee. Historical versions absent from
these subscription catalogs are unavailable through the configured routes,
not assumed retired globally. Additional paid API routes require separate
endpoint/model/auth/budget configuration; no purchase or paid fallback is used.

Google OAuth succeeded for the old Gemini CLI0.64.0, but the provider rejected
Code Assist access and explicitly requested migration to Antigravity. That
failure is preserved separately. Antigravity was separately authenticated and
its model catalog fetched. The native PNG cache matched the frozen image hash
for a real Gemini and a real Claude request. CLI1.4.2 exposes a broad init tool
list despite the custom agent whitelist: we disclose this and audit ACTUAL
executed tools, rejecting any action except the supplied PNG view. Native
conversation database media metadata and cached PNG SHA256 confirm image use;
backend serialization is unobservable. There is no repository/label/history
in the isolated request directory. Initial interface failures and recovered
original observations are retained; no result-driven replacement calls.

## GPU surrogate research lane

Eleven new conditions use the next two fixed dev IDs per direction (positions
2:4), disjoint from the commercial10 and final150; these are historically seen
surrogate dev images, not a new untouched holdout. Each uses10 images total,
48 references/class, BF16 parameters,224px, epsilon16/255,50steps, seed42.
One research GPU job is interleaved after two original dispatches, with at most
one research job alongside the unchanged original queue.

| ID | Change from U00 |
|---|---|
| U00 | A3 four-layer centroid margin, local0.35, DI, equal fusion; CLIP+Qwen2B+InternVL2B |
| U01 | Replace global margin with0.75/0.25 Pull/Push; retain local/layers/DI |
| U02 | Global margin only; local weight0 |
| U03 | Soft multi-reference affinity instead of centroids |
| U04 | Global0.5 margin +0.5 Pull/Push; retain local |
| U05 | PCGrad-inspired fusion |
| U06 | Surrogate lookahead adaptive weighting |
| U07 | Four EOT samples; explicitly expanded forward/backward budget |
| U08 | Replace Qwen2B with DINOv2-Large, fixed K3 |
| U09 | Add SigLIP2, K4 |
| U10 | Add DINOv2-Large, K4 |

Held-out native validators: Qwen9B, InternVL8B, Gemma E4B; none participate in
these attacks. They are distinct checkpoints, not necessarily independent
vision backbones/families. Selection uses minimum validator×direction ASR,
then pooled ASR, then measured compute. No commercial or final-target feedback
selects an algorithm. All outcomes including failures remain visible.

## Mechanism lane

Already executed CPU-only reference geometry:15 pinned models,480 aligned
reference images/model,105 cross-model pairs. Outputs include cosine
dispersion, covariance trace, effective rank, layer CKA, cross-model CKA/RSA,
within-class-residual CKA, and joint PCA/t-SNE. These are reference-only plots,
not attack displacement plots or a causal explanation of transfer.

New admitted GPU jobs extract frozen clean/adversarial four-layer features
for research proxies and validators. CPU jobs then compute per-image
DeltaPull/DeltaPush/DeltaR, margins, class geometry, cross-model CKA/RSA,
joint reference+clean+adversarial PCA/t-SNE and exploratory Spearman matrices.
All fits share one coordinate system across groups. t-SNE distances are not
quantitative evidence. With two images/direction, per-direction CKA is
degenerate and RSA undefined; they are N/A, while cross-model measures use
all10 aligned images. Repeated images limit statistical independence.
Reverse-direction asymmetry is unavailable until a reverse cohort is run.

Full artifacts remain on the persistent root disk under the V7 output root,
with this study in `proxy_composition_18h/`: `commercial_model_versions/`,
`surrogate_followup/`, `mechanism_reference/`, `unified_scheduler/`.
Only compact measurements, reports and a few example figures are published.

Source docs: [Google authentication](https://geminicli.com/docs/get-started/authentication/),
[Antigravity headless](https://www.antigravity.google/docs/cli/headless/),
[custom agents](https://www.antigravity.google/docs/subagents/),
[Codex CLI](https://developers.openai.com/codex/cli/reference).
