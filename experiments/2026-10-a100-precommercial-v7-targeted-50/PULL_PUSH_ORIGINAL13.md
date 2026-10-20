# Rerun the original13 Multiple Proxy rows with balanced four-layer Pull/Push

User explicitly selected the original13 unique Excel combinations on2026-10-20,
not the96-row composition expansion. This is a fixed-method test campaign,
not a new target-guided tuning loop. No commercial requests are authorized here.

All rows use four visual depths25/50/75/100%, uniform layer averaging,
spatial-token mean pooling, pure Pull0.5/Push0.5, no Patch/Margin/CLS/projector
loss, DI on, independently L1-normalized per-image proxy gradients averaged
equally,50 steps, epsilon16/255, step1/255, momentum1, random start and seed42.
Every open-model floating parameter is BF16 and every load is audited; no
quantization or precision fallback. Five original GT-selected directions,
30 images each, same48 references/class and same prompt/parser as prior runs.

| IDs | Proxy composition | Target |
|---|---|---|
| E01/E02/E03 | InternVL2B /2B+4B /2B+4B+8B | Qwen27B |
| E04/E05/E06 | Qwen2B /2B+4B /2B+4B+9B | Gemma31B |
| E07/E08/E09 | Gemma E2B /E2B+E4B /E2B+E4B+26B | InternVL14B |
| E10 | InternVL2B+Gemma E2B | Qwen27B |
| E11 | E10+CLIP-L/14 | Qwen27B |
| E12 | E11+SigLIP2 | Qwen27B |
| E13 | E11+DINOv2-Large | Qwen27B |

Physical execution order within each ensemble follows the previous actual A3
condition, preserving DI/RNG comparability. Each target's entire VLM family is
excluded from its attack proxies. Original repeated spreadsheet rows refer to
the same measurement. Total13 combinations×5 directions =65 cells,1950 rows.

Attack and target evaluation run in separate processes. All attacks in a target
group must be frozen before its evaluator starts. Each target loads once per
evaluation batch. Inference retains the historical clean/adv pair batch of two;
only model weights are reused, preserving the earlier evaluator's batching.
An additional immutable evaluation-runner lock records this grouping adapter.
Per-batch target metrics remain resumable.

One admission owner takes the same global GPU scheduling lock. At most4 GPU
workers,72GiB predicted GPU reservation,96GiB host-memory reservation, and
actual free-GPU/host checks. Large initialization peaks are reserved first.
After all proxy loads and the first vision forward, an audited runtime marker
permits a conservative steady-state reservation using resident vision weights,
prior measured activation peaks and4GiB extra headroom. Gemma language weights
stay BF16 CPU, with their actual CPU parameter memory accounted for. No proxy,
image, step or precision is dropped to fit memory.

Each cell saves frozen PNG hashes, IDs, actual layers, settings, gradients and
compute traces locally. Failures are recorded and independent work continues;
failed cells remain incomplete. The launcher never silently retries a failed
scientific cell with changed settings.

Compare unconditional target hits/150 and hits/30, paired image wins/losses and
image bootstrap intervals against compatible old A3 and historical V6 results.
Missing old measurements remain N/A. A3 changes global loss **and** patch relative
to this run; V6 additionally changes depth/DI. These are whole-method contrasts,
not coefficient-only causal evidence. Use the prior matched dev ablations for
component-specific comparisons. None of these final target outcomes selects
commercial settings.

Launch: `bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_pullpush_original13.sh`.
Outputs: persistent `outputs/a100_precommercial_v7_targeted50/pull_push_original13/`.
Compact reports publish under `reports/pull_push_original13/`; full PNGs, feature
caches and traces remain on the root disk. The historical V6/V7 folders are read-only.
