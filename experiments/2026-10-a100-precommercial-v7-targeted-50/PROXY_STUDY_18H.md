# Hypothesis-driven proxy composition study: 18 hours

This is a new, bounded study. Prior V6 and active V7 studies remain unchanged.
New code runs from an isolated worktree. Full PNGs/checkpoints use the persistent
root disk at `outputs/a100_precommercial_v7_targeted50/proxy_composition_18h/`;
attack PNGs retain the shared V7 hashed artifact convention with new P18 labels.

## Questions, not expected numerical conclusions

H1: At equal proxy count, heterogeneous families may outperform same-family
multi-scale proxies. Test C+A1+B1 versus C+A1+A2 and C+B1+B2 across three targets.

H2: Visual training families may contribute different transfer information.
Test C+D, C+S and D+S at K=2; then fixed A1+B1 with different visual pairs.
Different architectures/native resolutions are part of model identity, so this
does not isolate a causal training-objective effect.

H3: Additional proxies can be redundant or conflicting. Compare A1+B1 with
same-family pairs; C+D+A1/B1 with C+D+A1+B1; finally A1+A2+B1+B2. Record gradient
cosine, negative-pair fraction, cancellation and per-proxy optimization margins.

H4: More proxies may have diminishing benefit per unit compute. Report actual
forward/backward counts, runtime, VRAM, ASR gains and paired uncertainty. K is
not a substitute for compute. Four/eight-model conditions are optional extensions.

## Scope and priorities

The CSV has 32 templates x three targets = 96 candidate conditions. Core M01-18
has 54 conditions, with 22 completed reuse candidates at planning time. Full
matrix has 23 completed reuse candidates. Existing queued/partial conditions
resume their original artifact identities; no historical single/layer campaign
is restarted. Missing matched singles are limited to this composition study.

Fixed queue: same/cross-family paired contrasts first; visual-pair and mixed
contrasts next; all remaining core rows; extensions only if all core rows finish.
No queue order or hyperparameter changes based on final-target responses.
This budget does not guarantee 54 or 96 completed conditions. Publish exact
completion, partial checkpoints and failures at 18 elapsed hours. Rerunning does
not reset the original deadline. Last 45min admits no new 150-image attack;
last 15min admits no new model load. Interrupted microbatches rerun, committed
batches resume with PNG hashes. Individual errors do not stop independent rows.

## Frozen scientific settings

A3 centroid ten-class margin; four depths .25/.5/.75/1; patch weight .35;
DI; equal per-image L1 gradient averaging; 224px; eps16/255; 50 steps;
step1/255; momentum1; random start; seed42; 48 refs/class; same 5x30 GT images.
Every floating model parameter BF16, no quantization. All VLM proxies exclude
the final target's entire family. No target inference in attack generation.
Targets read only frozen hash-verified PNGs after method lock. Parameter audits
are repeated for each load. Old results require full configuration/hash matching.
New locks include evaluator, prompt/parser and backend hashes as well as attacks.

## Execution

`bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_proxy_study_18h.sh`

Add `--preflight` to validate paths, cohorts and the immutable plan without GPU work.
The same command resumes. One predictive pool admits at most FOUR total GPU
workers including the existing primary job. Global admission budget is72GiB;
individual process caps, actual GPU memory and primary stages are monitored.
Supplemental workers yield before primary target/analysis stages; no primary
process is signalled. Models exceeding simultaneous residency use existing
deterministic sequential BF16 offload; no silent model/dtype/sample reductions.

Expected new core execution:18-30h based on observed completed loops; this18h
window prioritizes complete matched contrasts and reports anything unfinished.
The full candidate matrix would require roughly40-70h, pending new throughput
measurements. These estimates exclude the current campaign and commercial work.

## Reporting and follow-up

Live report, completion/directional CSVs, paired stratified bootstrap intervals,
discordant-image counts, unadjusted exploratory McNemar/binomial comparisons,
dtype audits, calibration and admission logs remain inspectable. Missing scores
are N/A. A point estimate over30 images does not prove population ASR>=50%.
Existing target-seen cohorts make this exploratory; independent holdout and seeds
are required for robust confirmation, not a new target-driven method search.

While attacks run, independently audit historical raw outputs, labels, PNGs,
cohorts and precision. Then prepare THREE-image clean/adversarial CLI validation:
GPT via Codex, Gemini via Gemini CLI, Claude only through an actually supported
and verified CLI route. Client name is not model identity. No paid API fallback
without configured route/auth/model/spend. No commercial feedback tunes attacks.
