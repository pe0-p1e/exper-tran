# Ten-image GPT transfer check

User-authorized extension on 2026-10-19. Target goal: at least 6/10 targeted hits;
this is an objective, not a promised result or a stopping/selection rule.

Run `bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_commercial_ten.sh`.
The launcher freezes and validates the package, reuses six identical earlier
observations, and executes the remaining fourteen requests through the existing
ChatGPT-authenticated Codex CLI. No paid API fallback. Interrupted work resumes
valid observations without replacing negative results.

## Frozen design

- First two ground-truth development images from each T01/T03/T05/T07/T09,
  with reference/test ID disjointness and PNG hashes checked before requests.
- Three images overlap the previous pilot; seven have no previous commercial
  observations. Report ten-image and seven-new-image measurements separately.
- The method remains the historical surrogate-selected `T10_confirm_0`:
  CLIP-L/14 + Qwen2B + InternVL2B, A3 four layers, patch0.35, DI,
  equal gradient fusion, seed42, steps50, epsilon16/255, 224px.
- Use existing frozen PNGs; no GPT feedback changes generation, images or method.
- Model gpt-6.1-sol, client codex-cli0.163.0, reasoning effort low, unchanged
  ten-class prompt/parser, direct image attachment, fresh isolated process,
  no model tools. Backend serving snapshot is not exposed.
- Ten-image ASR is targeted hits /10, including invalid predictions as misses.
  Transport failures are missing observations; ASR is N/A until all pairs exist.
  Record clean accuracy and all five direction counts. Do not merge this
  exploratory small development sample with the final150-image benchmark.

## Diagnosis

Use saved surrogate step traces to report loss improvement, representation
margins and gradient conflict. Step50 traces describe the transformed,
pre-update image, not an independently measured final-PNG output or margin.
Positive proxy feature margins do not establish target decision-boundary
crossing. Closed-target internals are unavailable, so preprocessing/representation
mismatch and gradient conflict remain hypotheses, not established causes.

If below goal, preserve the result. Further optimization uses separate
surrogate validation and a new method lock, rather than repeated searches on
these ten GPT responses. Gemini and Claude remain separately blocked until
authentication and actual model IDs are verified.

Full package: `outputs/a100_precommercial_v7_targeted50/commercial_pilot10/`.
Logs/results/locks: same V7 root under
`proxy_composition_18h/commercial_ten_image/` on the persistent root disk.

## Executed result

All20 image requests were verified in distinct sessions, with14 new requests
and6 reused from the earlier pilot. GPT clean accuracy10/10; targeted hits4/10
(40%), below the6/10 objective. By direction: T01 1/2, T03 1/2, T05 0/2,
T07 2/2, T09 0/2. The seven newly observed images contributed4/7 hits;
this does not replace the full denominator10.

Saved step50, transformed pre-update margins were positive on10/10 images
for CLIP and InternVL2B, but only2/10 for Qwen2B. All three surrogate losses
decreased on all10. Across the saved trajectories, negative gradient-pair
fraction was0.0727 and mean cancellation ratio0.6042. These establish incomplete
Qwen surrogate feature-margin optimization, not a proven causal explanation
of the closed-target outcomes. There is no independent final-PNG margin replay
in this CPU-only diagnosis.

Report generation independently verifies raw structured target answers,
session uniqueness and PNG identities. One recovered historical observation
has no recorded start time, so overall request duration has19/20 coverage;
all14 newly issued request durations are recorded (54.5seconds total).
