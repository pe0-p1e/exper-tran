# Live Multiple Proxy supplement implementation

The current32-condition correction continues unchanged. The user's subsequent
request authorizes measured GPU concurrency, with at most FOUR GPU workers in
total. This supersedes the initial conservative single-owner waiting policy
only for the supplementary scheduler; it does not relax scientific isolation.

`run_dynamic_pool.sh` schedules one original owner plus up to three supplementary
workers. `run_multiproxy_supplement.sh` retains its sequential and one-sidecar
execution paths. There must be only one supplement scheduler: all use the same
`multiproxy_8h_supplement/scheduler.lock`. The active attack was handed from the
single-sidecar controller to the pool without stopping that attack or signaling
the original correction. The recorded handoff identifies both PIDs.

The fixed52-row CSV is deduplicated to47 target-specific conditions. Ten have
validated reusable answers at the first server inventory. The immediate missing
multi-proxy work is nine two-member subset conditions and eight additional
original-table compositions. Two missing Gemma matched singles remain explicit
pending items; this supplement does not relaunch a historical Single Proxy or
Embedding Layer table. The original32-condition queue, including its existing
loss/layer/seed controls, is never modified or canceled.

## Predictive memory admission

Before dispatch, `dynamic_pool.py` computes a reservation from actual audited
parameter bytes, measured microbatch activation peaks, and checkpoint-loading
peaks. Gemma's complete BF16 loading footprint counts even though its unused
language weights later move to CPU. Already running jobs keep their future
peak reservations, rather than only their current CUDA occupancy. The global
budget is72GiB, leaving8GiB on A10080GB. Each new child also receives its own
PyTorch allocator limit. Both GPU process count and actual global memory are
monitored; only supplementary children may be stopped by a guard. Completed
PNG batches survive such a yield. Precision, image count, loss and50-step budget
are unchanged. The current subset pool uses two-image batches for matched
stochastic preprocessing.

Frozen-PNG target evaluation has a separate memory admission. Its outputs do
not influence any later attack choice or configuration. The new scheduler
drains workers when needed to admit a large target. All supplementary workers
yield if the original owner enters its own target evaluation/analysis phase.
An equal number of GPU workers need not maximize throughput; utilization,
batch times, compute counts and memory are retained for evaluating that claim.

Supplementary model audits use a separate stream from the original correction,
with a writer lock among pool children. Shared immutable manifests and the
active global model-revision audit are read without rewriting them.

## Extension preparation and evidence

Four additional checkpoints are pinned in `supplement_model_revisions.json`:
CLIP-B/32, CLIP-B/16, CLIP-L/14@336 and DINOv2-Base. CLIP-L/14 and DINO-Large
already have independent pinned identities. The additional checkpoints have
downloaded; their real CPU BF16/gradient checks and later A100 checks are
separately labeled. A CPU result never counts as an A100 smoke or target ASR.
The336 CLIP has its own differentiable336px preprocessing and cache identity.
DINO's checkpoint config declares518, while its processor declares resize256
and crop224. The V5-compatible DINO path here explicitly uses square256 without
center crop; it is not described as an exact reproduction of the default
processor. CLS/padding are excluded from spatial pooling.

The optional CLIP4/DINO2 families are not automatically launched. Required
existing compositions take priority. Real gradients, reference extraction,
resource calibration and surrogate-only dev selection must precede any new
optional full-cohort evaluation.

## Persistent outputs and reporting

Everything is under `outputs/a100_precommercial_v7_targeted50/multiproxy_8h_supplement/`:

- `reports/multiproxy_8h_server_inventory.csv`, `multiproxy_8h_queue.csv`,
  `multiproxy_8h_results.csv`, `multiproxy_8h_report.md` and three matrix-specific
  completion tables;
- `results/per_image_results.csv` and paired-comparison data;
- `pool_state.json`, `pool_runtime_lock.json`, immutable method locks and original
  eight-hour `budget.json` (restarting does not reset that window);
- `audits/*_pool_admission.json`, `dynamic_memory_benchmark.json`, full BF16 load
  events, model gradient/calibration records and CPU-preflight evidence.

The commercial three-image blinded pack is preserved. CLI access was inspected
without inference: Codex is signed in, Antigravity is not signed in. Exact fixed
commercial model identity/image-session isolation and paid spend settings remain
unverified. No commercial inference has been executed by this supplement.

Use COMPLETE/RUNNING/PARTIAL/BLOCKED/QUEUED honestly. Missing ASR remains N/A.
Only compact reports/measurements/provenance are uploaded; complete PNGs and
feature caches remain on persistent root disk, checkpoint weights on ephemeral
cache. This file describes execution, not a claim that the campaign is complete.
