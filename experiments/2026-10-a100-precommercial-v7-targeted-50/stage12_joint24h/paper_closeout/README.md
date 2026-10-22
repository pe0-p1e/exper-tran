# Final paper closeout: user-authorized2026-10-21 scope

Single entry point, already authorized to execute:

```bash
bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_paper_closeout.sh --start --deadline-hours 24
```

`--resume` preserves the first closeout deadline and valid results; `--status`
does no inference; `--report` computes/publishes current data. Stop through the
existing `run_stage12_joint24h.sh --stop`. This is an extension of ONE existing
GPU admission owner, not a competing scheduler. The old window is archived and
preserved; `closeout_window.json` records the newly authorized scope/deadline.

## Contracted experiments

1. Fixed same3 legal proxies: independently top-ranked4 layers, FDA-inspired
   conditional3 and4 layers. The six development10 depth candidates and ranking
   are reused. Every joint subset generates its own attack; no union of singleton
   successes. Complete matched explore30 and confirm50 with identical scientific
   settings and image IDs. Existing matching conditions/PNG hashes are reused.
2. GPT6.1 Sol low development50 compares7 matched three-proxy candidates:
   prior selected2/3 per family, independent4, FDA2/3/4, Uniform2/4. Qwen-style
   legal proxy pool is predeclared for generic commercial packets. Previously
   returned identical clean/adv/model/prompt/client records are reusable.
3. **User explicitly changed the selection rule:** use these GPT50 results to
   pick one selected strategy, one FDA strategy and one Uniform strategy; ties
   prefer fewer tapped layers, then stable name. Thus GPT50 is development and
   target-feedback-assisted selection, NOT untouched confirmation. Each candidate
   and failed request remains visible. No final150 output can change this lock.
4. New disjoint150 GT images,5 directions×30, same images for every method.
   Freeze3 strategies, apply their predeclared family-specific legal proxy/layer
   maps, evaluate Qwen27B,InternVL14B,Gemma31B. Commercial evaluation uses the
   same predeclared generic qwen-pool packets across GPT6.1 Sol,6 Sol,5.6 Sol,
   ALL low. No Astra. Up to1800 unique commercial150 requests before exact
   artifact deduplication (150clean+3×150adv per model). API/snapshot identity
   beyond the actual CLI requested/returned evidence is not invented.
5. Separately redo5 Pull/Push ratios:1/0,.25/.75,.5/.5,.75/.25,0/1. Fixed3-proxy
   **Uniform4**, same explore30 IDs,50steps,16/255,alpha1/255,momentum1,DI,seed42,
   BF16, equal independent per-image L1-normalized gradients. Each large target
   gets its legal family-excluded pool; GPT6.1 Sol low uses the predeclared qwen
   pool. Report hits/N and unconditional ASR; legacy conditional TASR is distinct.
   Ratios cannot change final150 methods. No other large ablation or historical
   Single/Layer/Multiple table rerun is admitted by this scope.

## Execution and precision

The existing8-worker maximum remains an upper bound; actual free VRAM/RAM,
calibrated reservations, cgroup memory limits,72GiB aggregate cap and8GiB/10%
VRAM headroom decide admission. Full BF16, unquantized model parameters are
recursively audited before inference. CPU language offload is only for unused
proxy weights; large target inference loads directly on GPU. No CLS/logit loss.
Mean per-layer loss within each model, then normalized equal proxy gradients.

GPU attack/open-target workers, CPU-only commercial requests and CPU analysis run
in parallel. The tested isolated subscription CLI handles each image in a fresh
context with tools/network disabled, strict parser, raw trace capture and no paid
fallback. At most one actual commercial request at a time is initially used;
chunked20-request progress provides fairness and resumes cached requests. Quota/
authentication failures block instead of rotating accounts or bypassing limits.

Every image has an atomic PNG receipt/hash/bound check. Equivalent scientific
conditions reuse the same PNGs across models; different loss/layers/cohort/seed
cannot reuse attacks. At one hour before the new deadline no new GPU/CLI work is
admitted; owned workers checkpoint. Analysis/publication completes the final
half-hour. Partial/failure never becomes zero or COMPLETE. A disconnected
supervisor handles ordinary retries; code repair still needs an active agent.

## Measurements and publication

Permanent root disk:
`outputs/a100_precommercial_v8_joint_selection24h/paper_closeout/`.

Native target-deepest spatial embeddings are extracted from the already loaded,
audited evaluator using the same native differentiable visual adapters, after
PNGs freeze. Cache48 references/class and clean/adv vectors by pinned revision/
PNG hashes. No target representation enters attack generation. Compute DeltaR,
pull/push, margins, gap closure, prototype distance, cosine dispersion, covariance
trace/effective rank; aligned-reference CKA/RSA per model-layer/target, plus
explicit descriptive mean across model-layer pairs. Raw embeddings are exported
so alternative aggregation remains possible. Closed GPT weights/embeddings are
unavailable; its representation shift/CKA/RSA must be N/A.

PCA/t-SNE fit a SINGLE normalized union of both reference classes and the available
paired clean/adversarial images; PCA40 preprocessing, t-SNE perplexity30/initPCA/
seed42. This campaign has forward directions only: absent reverse attacks are
explicitly absent, not synthesized. t-SNE distances are qualitative. Spearman
matrices include pairwise observation counts; shared-image/model dependence and
development selection preclude a causal/equivalence claim from these plots.

GitHub compact data path:
`experiments/2026-10-a100-precommercial-v7-targeted-50/reports/v8_paper_closeout/`.
Periodic commits/push include complete per-image/direction outcomes, configurations,
all available per-step numeric attack receipts in deterministic compressed chunks,
BF16 load audits, target/source embeddings, projection coordinates and figures,
historical hyperparameter report datasets and SHA manifests. Full PNGs, model
weights, authentication and raw CLI sessions remain on permanent local storage.
Missing old measurements are labeled missing; old experiments are not secretly
rerun or relabeled. Publication failures retain local output and explicit error.

`reports/FINAL_REPORT.md` is an actual-results readout, not a manuscript draft.
Final report contains completed/partial cells and all compared candidates; it must
not call150 complete if responses are missing. Selection cost, per-image measured
walltime and compute counters require concurrency-aware interpretation.

## Latest user override: run until completion, no automatic deadline

The user revoked the tomorrow cutoff. `run_policy.json: run_until_complete=true`
now disables every scheduler, supervisor, commercial and analysis deadline guard.
The original time windows remain historical estimates; status reports
`automatic_deadline=false`, `remaining_hours=null`. Do not substitute another
hidden deadline. Complete the frozen scope and archive automatically; explicit
user stop and genuine reported failures still take effect.

```bash
bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_paper_closeout.sh --resume --until-complete
```

This resumes the same receipts, request IDs and single admission owner. Changing
this execution policy does not change scientific settings or add experiments.

## Phase-aware physical host limits

Loading reservations are no longer held for an entire worker lifetime. After a
real gradient step or the evaluator's post-load validation heartbeat, the owner
reads cgroup anonymous/kernel memory and retains its observed high-water value.
It lowers the actual verified worker cgroup limit to at least8GiB or
1.25×that nonreclaimable high-water+3GiB, never above the original load limit.
Only successful privileged cap updates reduce admission accounting. File cache
is reclaimable; virtual address space is never used as a RAM limit.

Full-GPU BF16 proxy placement can also be chosen under logical host reservation
pressure, even when system MemAvailable is still high. Its host loading bound
uses the largest sequentially initialized model×1.4+5GiB and measured dedicated
full-GPU loading peaks, rather than summing CPU language weights that are not
retained there. CPU allocator pages are released after each fully moved model.
Full BF16 weights, references, per-image RNG and attack math are unchanged.
The controller adopts live workers on update; physical headroom,72GiB total
GPU reservations and maximum8 workers remain enforced. Higher concurrency is not
a claim of proportional speedup; compare actual image completions and runtime.

## Transport recovery and final audit (2026-10-22)

The isolated CLI adapter waits up to600s per request and retains partial timeout
stdout/stderr. An unscored timeout can be retried at most3 times, with each failed
attempt archived by request ID. A returned answer/completed turn requires manual
reconciliation; successful result.json records are immutable and never resent.
Unknown failures and quota/authentication errors are not automatically retried.
No model, image, prompt, low effort, parser, attack or paid route is changed.

`scripts/finalize_v8_statistics.py` independently checks canonical evaluator and
CLI records, fixed cohorts, decoded PNG pixel bounds, recorded BF16 events,
reference/attack IDs, numerators/denominators and request image/prompt hashes.
It produces completion_matrix.csv, confidence_intervals.csv (Wilson), exact
paired McNemar tests with within-target Holm correction, direction-stratified
paired-bootstrap intervals, and FINAL_EXPERIMENT_REPORT.md. Non-significance
is not equivalence, and correlated target-image rows are not pooled as replicates.

The CPU-only `scripts/watch_v8_final_closeout.py` monitors the already running
commercial driver and refreshes these checks. After completion (or an explicit
bounded external blocker), it drains only its verified analysis service, performs
a final report/publish, and archives the full output tree and source snapshot to
`/home/ubuntu/experiment-checkpoints/` on `/dev/vda1`. It never starts GPU work.
The full PNG corpus and raw CLI traces remain local; GitHub receives compact
numerical results and reports. PARTIAL is never labeled COMPLETE.
