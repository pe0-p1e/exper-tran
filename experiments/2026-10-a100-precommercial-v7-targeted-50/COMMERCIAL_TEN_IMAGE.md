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
