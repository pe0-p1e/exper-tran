# Commercial validation: method-selection provenance

## Existing three-image pilot

The pilot uses the previously locked `T10_confirm_0` method, not an arbitrary
proxy combination. The source is
`outputs/a100_precommercial_v7_targeted50/ten_hour_research/selected_method_lock.json`.
The lock records selection using held-out surrogate native outputs only and
`final_target_feedback=false`.

At seed 42, the development confirmation CSV records these three candidates,
each with eight images per direction:

| Candidate | Proxies | Recorded minimum native ASR | Recorded pooled native ASR |
|---|---|---:|---:|
| T10_confirm_0 | CLIP-L/14 + Qwen2B + InternVL2B | 0.750 | 0.9667 |
| T10_confirm_1 | Qwen2B + InternVL2B | 0.625 | 0.9167 |
| T10_confirm_2 | CLIP-L/14 + DINOv2-Large + InternVL4B | 0.750 | 0.9417 |

These are recorded development aggregates, not commercial success rates or a
fresh independent recount. Their meaning follows the original development
evaluator. The chosen candidate wins among these seed-42 confirmation rows
by minimum score, then pooled score. Seed-43/44 runs are robustness checks,
not permission to replace the declared seed using whichever scores highest.

The attack uses A3, four visual depths, patch weight 0.35, input diversity,
equal normalized gradient fusion, 50 steps and epsilon 16/255. GPT was
explicitly invoked with `model_reasoning_effort="low"`. Low reasoning effort
does not imply that reported reasoning-token usage must be zero.

The three frozen pilot cases are the first fixed development image in T03,
T05 and T09. All six GPT clean/adversarial requests have already been executed:
clean accuracy 3/3 and adversarial target hits 0/3. Preserve those observations.
They do not establish that the selected method is optimal for GPT or that all
representation attacks fail on GPT.

## Next method freeze

The active 18-hour composition study has additional candidate ensembles. Its
final-target test scores must not choose the next commercial attack. Before
claiming a new best method:

1. Compare a predeclared, manageable set on separate surrogate development
   images, with held-out validators absent from the attack ensemble. Keep
   validation image IDs, model revisions, precision and compute explicit.
2. Use one common validation set and declared budget. Select by minimum
   direction/validator success, then pooled success, then measured compute as
   a tie-breaker. Record missing validation as missing, never zero.
3. Save candidate results, selection rule, source hashes and method lock before
   new commercial requests. Do not infer commercial optimality from open final
   target ASR or from optimization losses alone.
4. Prepare a distinct, hash-locked package if the method changes. Do not
   overwrite the existing package or reuse its request IDs for changed PNGs.
   Evaluation of a revised method on the already observed three cases is an
   exploratory paired comparison, not untouched final validation. Any future
   independent final cohort must be selected without prediction filtering.
5. Keep GPT reasoning effort low for matched comparisons. Keep provider model,
   client, prompt, image transport and isolation fixed and audited. Gemini and
   Claude require completed authentication and verified model IDs first.

No new commercial calls, new image cohort, GPU jobs, or modifications to the
running method locks are introduced by this provenance note. The existing
pilot remains a connectivity and limited transfer test; the next method is
not yet selected against all newly proposed candidates.
