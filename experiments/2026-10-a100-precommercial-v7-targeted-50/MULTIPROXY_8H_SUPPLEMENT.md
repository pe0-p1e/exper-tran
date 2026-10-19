# 2026-10-19 Multiple Proxy supplement: actual server status first, eight-hour GPU budget

## Binding current-user scope (overrides 2026-10-19 full-workbook rerun plan)

- **Do not launch a fresh historical Single Proxy campaign. Do not rerun the standalone Embedding Layer Sweep.** Those are established methodological/feature-depth evidence; optional matched-member singles already inside the running, frozen family-disjoint protocol may finish without cancellation and are distinct from redoing the two historical Excel tabs.
- Focus on the original **two Multiple Proxy tables**: 13 unique legacy configurations; the frozen **21-row target-family-disjoint 7-subset selection** matrix; optional **18 predeclared cross-family CLIP/DINO extensions**. The accompanying [multiproxy_8h_supplement_matrix.csv](multiproxy_8h_supplement_matrix.csv) has **52 candidate conditions, NOT 52 required new runs**. The 13 legacy and 21 selection conditions overlap; perform an exact hash-based union, never a blind sum. Reuse fully matching attack & target-evaluation artifacts.
- **Do not restart, change, interrupt or duplicate the currently running 32-condition/160-cell family-disjoint correction** (`FAMILY_DISJOINT_RECOVERY.md`). Already completed Gemma31B cross-family A3 results must be reused if fingerprints match (130/150=86.67% for CLIP-L/14+Qwen2B+InternVL2B; Qwen2B single 74/150=49.33%). Older Qwen/InternVL A3 results incorporating same-family VLM proxies **do not satisfy strict family exclusion**.
- The **new-loss ablations are reported to be already in progress by the user**. Inspect real server state and resume existing locks; **do not spawn any duplicate loss/weight/patch/no-DI/100–200-step ablation** in this eight-hour supplement. Finish current running work first without altering the lock. The earlier Pull/Push ablation is later.
- Eight hours is a **soft work budget / hard reservation limit**, not a guarantee that all candidates can finish, especially if other GPU jobs occupy the device. Prefer complete, validated 5-direction conditions to wide partial coverage. Never fabricate or silently treat pending as zero.
- After the supplement, **prepare/inspect** the commercial 3-image blinded pilot; execute only if actual verified image-capable fixed model routes, credentials, spend limits and user's approved execution exist. See `COMMERCIAL_PILOT.md`.

## The original 13 unique Multiple Proxy combinations (must appear in results table)

Use original 5 directed pairs T01,T03,T05,T07,T09; each is 30 frozen GT source images, 48 disjoint reference images/class. The new A3 method is 4 normalized visual depths (25/50/75/100%), 10-class reference margin + local patch weight0.35 + DI, normalized equal gradient fusion, 50 steps, eps16/255, step1/255, momentum1, random start seed42, BF16 all open model floating weights, no quantization, no target feedback during attack. Track new A3 results separately from the V6 deepest Pull/Push results.

| Original ID | Real target | Original proxy list | Status action |
|---|---|---|---|
| E01 | Qwen27B | InternVL2B | New A3 only if not already valid/in flight |
| E02 | Qwen27B | InternVL2B + InternVL4B | New A3 if missing |
| E03 | Qwen27B | InternVL2B + InternVL4B + InternVL8B | New A3 if missing |
| E04 | Gemma31B | Qwen2B | Compatible completed A3 74/150 on archived cohort; check hashes |
| E05 | Gemma31B | Qwen2B + Qwen4B | New A3 if missing |
| E06 | Gemma31B | Qwen2B + Qwen4B + Qwen9B | New A3 if missing |
| E07 | InternVL14B | Gemma E2B | New A3 if missing |
| E08 | InternVL14B | Gemma E2B + Gemma E4B | New A3 if missing |
| E09 | InternVL14B | Gemma E2B + Gemma E4B + Gemma 26B-A4B | New A3 if missing |
| E10 | Qwen27B | InternVL2B + Gemma E2B | New A3 if missing; overlap with subset AB |
| E11 | Qwen27B | InternVL2B + Gemma E2B + CLIP-L/14 | New A3 if missing; overlap with subset VAB |
| E12 | Qwen27B | E11 + SigLIP2-So400m | New A3 if missing |
| E13 | Qwen27B | E11 + DINOv2-Large | New A3 if missing |

These keep the workbook's two original blocks. Duplicate display rows refer to the **same measured condition**. Do not compute twice. Note that E01–E13 all exclude their respective target families, but a reused run must also have exactly the NEW A3 loss, reference banks, native preprocessing, seed, generation parser, cohort and model BF16 revision; old V6 Pull/Push (even BF16) is only a baseline, **not** completed new A3.

## The existing 21-row target-specific 7-subset table: highest reuse priority

Slots (V,A,B) and all nonempty subsets, with a single frozen exact A3 implementation:

| Target | V | A | B |
|---|---|---|---|
| Qwen27B | CLIP-L/14 | InternVL2B | Gemma E2B |
| InternVL14B | CLIP-L/14 | Qwen2B | Gemma E2B |
| Gemma31B | CLIP-L/14 | Qwen2B | InternVL2B |

For **each** target run/fill V, A, B, V+A, V+B, A+B, V+A+B = 7 rows, 5 directions per row. Do not rerun covered matched singles being produced in `family_disjoint_recovery`, nor the frozen primary multis. Compatible E04/E10/E11 from the legacy table may map onto selection slots where target, set, budget, and exact loss match. Re-evaluate completed artifact identity before reuse.

Repository snapshot 2026-10-19: selection table reports only Gemma31B A=74/150 and V+A+B=130/150 as complete. Qwen/InternVL entries and other Gemma subset entries are **pending in that Git commit**, while some attack-only artifacts show partial existence. This is **not proof of current server state**. Query server processes, campaign locks, output directories and per-cell hashes first.

## Additional vision-heavy family-disjoint candidates — optional, NOT a combinatorial sweep

Interpret the requested '4 CLIP + 2 DINO + VLM' as **FOUR different CLIP checkpoints + TWO different DINOv2 checkpoints + one or two cross-family VLMs**. Do not duplicate a CLIP checkpoint four times or introduce target family for intra-family shortcuts.

CLIP4 proposed actual Hugging Face IDs (pin exact verified revisions on first load):
- `openai/clip-vit-base-patch32` (CLIP-B/32)
- `openai/clip-vit-base-patch16` (CLIP-B/16)
- `openai/clip-vit-large-patch14` (CLIP-L/14; already used)
- `openai/clip-vit-large-patch14-336` (CLIP-L/14@336)

DINO2 proposed verified IDs:
- `facebook/dinov2-base`
- `facebook/dinov2-large` (already used)

Per-target VLM slots:
- Qwen27B target: InternVL2B + Gemma E2B. **No Qwen of any size**.
- InternVL14B target: Qwen2B + Gemma E2B. **No InternVL of any size**.
- Gemma31B target: Qwen2B + InternVL2B. **No Gemma of any size**.

**Six candidate variants per target**, not all mandatory to fully run:

| Candidate suffix | Composition | Number of proxies | Role |
|---|---|---:|---|
| CLIP4 | 4 distinct CLIPs | 4 | CLIP-only control |
| CLIP4_DB | CLIP4 + DINOv2-Base | 5 | Base contribution |
| CLIP4_DL | CLIP4 + DINOv2-Large | 5 | Large contribution |
| CLIP4_D2 | CLIP4 + DINOv2-Base + DINOv2-Large | 6 | DINO pair contrast |
| CLIP4_D2_A | CLIP4 + DINO2 + target-allowed VLM A | 7 | VLM incremental value |
| CLIP4_D2_AB | CLIP4 + DINO2 + both allowed cross-family VLMs | 8 | Full user-requested family-diverse ensemble |

18 optional variants ×5 directions = **90 additional cells, 2,700 image-level target evaluations if ALL are run**. Do **not** assume eight hours suffices. Each composition incurs surrogate backward passes per proxy, repeated 50 times. The huge 8-proxy `CLIP4_D2_AB` is a resource-heavy exploratory endpoint, **not** automatically the best. Prefer a smaller comparable subset with one CLIP, one DINO and both cross-family VLMs as a compute-matched auxiliary control if eight-proxy runtime is poor (document separately).

**Native resolution and support:** CLIP-L/14@336 needs native 336 input while final perturbation canvas stays 224; differentiable resize/preprocessing must allow gradients to reach original pixels. DINOv2 native processor can use larger image size; verify patch masks/token counts. Use actual deepest/normalized depths after inspecting configs; do not reuse CLIP-L/14 reference caches for B/16 or @336, or DINO-L caches for DINO-B. The current V7 `vision_features.py` supports families `clip` and `dinov2` but the four new checkpoints **are not registered** as pinned names in the existing model lock. Before any pilot, extend registry, verify exact HF revisions, real BF16 weights/vision taps, output shape, image-to-pixel gradient, and safe loading; if a specific candidate is unsupported, record blocker and skip only that exploratory variant, not existing committed runs.

## Runtime evidence and eight-hour allocation

- V6 BF16 full 13-ensemble Multiple Proxy attack gradient time **5.59 h**, full workflow **10.43 h**. It used original deepest Pull/Push, not A3.
- V7 complete `CLIP-L/14+InternVL2B+Gemma E2B` Qwen27B A3 attack: 2,274 s (~38 minutes) based on prepared selection diagnostic; other models/conditions vary. This is **attack generation**, not target inference or audit.
- The 32-condition currently running strict family-disjoint campaign (160 cells) may consume a considerable portion or all of the new eight-hour window. Never count historical 5.59 h as a promise of new A3+8-proxy throughput.

Proposed priority/time envelope **relative to the actual server start**:
1. **0:00–0:20** read status, enumerate all GPUs and processes, validate actual in-flight lock and artifacts; CPU-only prepare queue; no duplicate launches.
2. **0:20–2:30** let existing owner keep GPU and prioritize completion of its frozen 32 conditions; where appropriate and hardware permits, prepare pending subset configs and download/cache pinned vision models without GPU disruption. If owner still holds only GPU, do not launch competing heavy work.
3. **2:30–5:30** run *missing* cells for the **21 predeclared selection rows** and then missing A3 original E01–E13 cells; batch by common surrogate weights/reference cache to reduce reloads. This is a priority, not a deadline guarantee.
4. **5:30–7:15** run 2-image-dev smoke/gradient audit for CLIP4/DINO2 additions; choose only a small prespecified affordable selection by surrogate-only development evidence. If remaining time allows, fully test 3 target-specific full `CLIP4_D2_AB` rows or a smaller matched design, five directions×30 images each; do not launch all 18 unless calibrated completion time demonstrably fits.
5. **7:15–8:00** allow running batch to finish if it meets budget, freeze new PNGs, evaluate missing target outputs, validate/collate exact rows, write logs/reports; prepare commercial pilot inputs/routes without consuming unrelated GPU. No new long jobs during last 45 min unless independently measured ETA fits.

These slots are **soft**. Instead dynamically allocate remaining time by `estimated_wall_time = per-image-per-step measured cost × remaining_images × steps + offload + evaluation + safety buffer`; update after every batch. If GPU is occupied, time is consumed but no access is forced. Stop gracefully at ~7h30 for new attacks (reserve 30min for audits), with partial progress preserved. Do not kill other owners, silently preempt work, overwrite locks or launch two processes both believing they own the single GPU.

## Mandatory actual-server status audit, coordination and memory use

Codex on the server must run:
- `hostname; date -Is; git status --short; git branch --show-current; git rev-parse HEAD; nvidia-smi -L; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv`;
- `nvidia-smi` for per-process PID/VRAM; `ps -eo pid,ppid,user,etime,args` filtering campaign/attack/evaluate; `tmux ls`, `screen -ls`, optional `fuser` for GPU/lock ownership; `flock` nonblocking check; process command lines and output root;
- JSON `state.json`, `current_stage.json`, `method_lock.json`, per-direction/batch `state.json`, `metrics.json` and last logs for `family_disjoint_recovery`, `proxy_selection`, `ten_hour_research`, and any active ablation roots; verify hashes and correct target. Files in GitHub can be stale; don't infer live job status from commits.
- Create `reports/multiproxy_8h_server_inventory.csv` and `reports/multiproxy_8h_queue.csv` with status `complete_valid`, `running_owned_elsewhere`, `waiting_for_GPU`, `blocked`, `queued`, `partial_resumable`. Never label a row complete from an attack-only state lacking required target evaluation.
- A real family-disjoint gate must check `model.family != target.family` at EVERY launch and target evaluation; generic vision-only `clip`/`dinov2` are exempt from MLLM-family clash but still must pass BF16 and image-gradient checks.

**GPU ownership:** Existing tasks have priority. If only one GPU, no simultaneous heavyweight second experiment when first already saturates GPU. If more GPUs appear, assign distinct `CUDA_VISIBLE_DEVICES` devices to noninterfering processes; never assume a second GPU is available from host CPU count. On a shared GPU, require isolated reservations/explicit scheduler permission and spare memory with measured throughput; using every free MiB while slowing active jobs is NOT efficient throughput.

**Dynamic precision/memory:** all floating model weights BF16, no quantization; gradients and loss FP32 allowed. Use measured `max_memory_reserved` and `max_memory_allocated`, per-condition short gradients, try microbatch2 then 4 only if measured safe; use 1 for OOM. Keep at least max(8GiB, 10% of device VRAM) of headroom and be below repository's 75GiB reserved cap on A100-80GB. Use `torch.cuda.mem_get_info()`, `torch.cuda.memory_stats()`, per-PID usage and empirical memory before selecting batch. A new concurrent workload may require a smaller batch even when its standalone test succeeded.

A robust 8-proxy run should keep only visual towers needed for the surrogate loss resident where possible; move other BF16 weights to CPU, reuse/dispose individual surrogate activations sequentially, accumulate each model's **independently per-image L1-normalized** pixel gradient in FP32, and fuse **after all models**. Offload large models between steps if necessary. Precompute 48/class reference feature banks by model/revision/layer/image hashes, checkpoint per condition/transition/microbatch, lazily load the target only for post-freeze evaluation, group target inference per target to minimize repeated 27B/31B/14B reload. Do not silently alter input image count, epsilon, steps, reference bank, actual layer taps or target prompt for memory saving.

Cache evidence: attack state model identities, exact HF revision, full loss config (A3 patch=0.35 etc.), gradients, seed, manifest hash, PNG hash and target eval parser are REQUIRED to determine if a reused artifact matches a new row; duplicates are skipped only if ALL match.

## Required report and next commercial step

Produce:
- `multiproxy_8h_server_inventory.csv`: GPU/process/locks, active stage, most recent checkpoint and remaining live work.
- `multiproxy_8h_queue.csv`: all original13 + selection21 + optional18 with live status, estimated duration, GPU owner, priority, action and selected reason.
- `multiproxy_8h_results.csv`: per target, ensemble, five directions, per-direction targeted hits/30, pooled hits/150, clean-correct ASR, full proxy list and loss version, model weight audits, exact SHA/model revisions, source provenance, compute/runtime, complete vs incomplete.
- `multiproxy_8h_report.md`: complete/partial/failed/blocked jobs, whether original two Multiple Proxy tables have matching new-loss results, new vision-heavy ensemble pilot/full results, marginal gain vs strongest matched single and 2/3/4/6/8 proxy alternatives, possible overfit/leakage, failure classes, time consumed and recommended remaining work. Do not claim an 8h success if GPU was blocked.
- Don't alter the user's original Excel layout or replace conditional TASR with ASR; produce mapping CSV for future filling.

After completing open-weight work, prepare the **three-image blinded commercial CLI pilot** (not a broad costly test). Preflight exact OpenAI/Gemini/Claude image-capable model IDs and CLI routes, authentication, isolated sessions, raw answer capture, model snapshots, PNG/prompt hashes and cap; 3 images ×(clean+adv)×3 target models=18 image-model evaluations. If budget/access missing, state blockers and stage the pack. Keep commercial outcomes strictly post-freeze; never use them to tune final method then report zero-query transfer on the same images.
