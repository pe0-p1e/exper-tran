# V8: family-disjoint joint model–layer study

User authorized immediate execution on2026-10-20. Worktree `/home/ubuntu/exper-tran-v8`; branch `experiment/a100-precommercial-v8-joint-selection-24h`. Historical V6/V7 scientific files and results remain unchanged.

Single entry point (already started):

```bash
bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_stage12_joint24h.sh --start --deadline-hours 24
```

`--resume` preserves the original absolute deadline and valid image/evaluation receipts. `--status` reads status without a GPU lock or inference. `--report` rebuilds compact evidence tables and handoff. `--stop` gracefully stops owned work; valid images remain. `--preflight` performs data/environment checks without model inference. `--smoke` runs calibration/two-image attack/native validation gates only. `--stage2-only` requires an existing immutable method lock and frozen50-image attacks, and uses the same supervisor/lock path; `--resume` automatically enters Stage2 once its prerequisites pass. No manual stage command is needed.

Supervisor: detached `tmux` session `v8-stage12`, independent of SSH/browser. It holds the SAME shared admission lock used by the historical PP4 scheduler. Workers inherit the lock descriptor; controller restart adopts only exact PID/starttime/command leases, not arbitrary CUDA processes. Unexpected controller exits get up to3 restarts. A machine reboot still requires starting the launcher again: no systemd/linger installation is claimed.

Output root is the permanent root disk:
`/home/ubuntu/exper-tran/outputs/a100_precommercial_v8_joint_selection24h/`.

## Fixed science

Pure `.5(1−cos(z,mu_target))+.5(1+cos(z,mu_source))`;48 references/class; native model-specific block IDs; spatial-token mean and L2 pooling; strict unquantized BF16 model parameters;50 steps;224px;L∞16/255;alpha1/255;momentum1;DI;seed42. Mean layer losses inside each proxy, independently L1-normalize image gradients, then equally fuse proxies. FP32 image gradients/statistics. No language-head loss, ten-class margin, patch objective or adaptive weights.

Each actual held-out validator excludes its entire family from the attack. Qwen9,InternVL8,GemmaE4 use separate legal proxy pools. Final Qwen27/InternVL14/Gemma31 and the150-image formal test remain out of selection. Multiple validator observations on common images are correlated.

V8 uses per-image hashed RNG for initialization/DI and individual-image forwards. This gives scheduling/partition invariance but differs from historical batch-shared RNG. Historical PP4 artifacts are continuity references only, not silently reused as identical V8 attacks. Existing hash-verified reference feature banks and model weights are reused where compatible. Model order is explicit in V8.1 condition identity. Early v8 smoke-only artifacts and the source-lock amendment remain archived; no50-step result preceded that amendment.

## Bounded execution

1. New GT-only exploration30 and confirmation50, disjoint by historical ID and decoded canonical-pixel hashes. Historical dev10 remains screen/debug data. Raw ImageNet source classes expanded from112 to320 images using the existing downloader. Stage2 reuses the confirmation50 only after method freeze.
2. Calibrate batch1/2 and run two-image/two-step attack/native-validation smoke for all three validator families. Scientific workers use individual-image microbatches, so fallback does not change per-image RNG or batch numerics.
3. Fixed-four-layer proxy composition controls, six native depth candidates for each of three backbones/family, conditional and width2 beam joint model–layer candidates, within-backbone FDA-inspired greedy2/4 plus uniform/top-independent/low-CKA controls. Weak-but-different candidates are retained through reference CKA screening. No faithful FDA auxiliary-classifier or ILA displacement-loss reproduction is claimed.
4. At most270 condition identities, including lower-priority bounded searches. Top≤3 per family enter30-image exploration; selected and uniform enter50-image confirmation. Selection uses development evidence and predeclared5pp tolerance; confirmation cannot retune it.
5. Freeze ONE selected method and matched uniform for each family BEFORE confirmation. Generic GPT method is predeclared to use the Qwen-dev family selection; no choosing the best family from GPT outputs.
6. Stage1 completion automatically starts GPT-6.1 Sol low: balanced10-image gate then expansion to the SAME50, selected and uniform. Up to150 unique requests (50clean+100adv), fewer only when artifacts are genuinely identical. Existing subscription CLI isolation/parser reused through an isolated50-image adapter. No paid API/model substitution. Raw traces, failed attempts and incomplete denominators preserved.
7. Stop admitting new search at T−6h and reserve Stage2/report time. If Stage1 prerequisites are incomplete, report the blocker rather than falsely freeze a method. At T−1h stop additional CLI admissions and audit/report remaining partial work. Timebox is not a guaranteed complete matrix.

## Recovery and monitoring

Predictive reservations reuse `run_proxy_study` historical weight/activation evidence and actual V8 calibration/peaks. Admission additionally requires actual free VRAM and max(8GiB,10%) headroom, aggregate≤72GiB, host RAM and no more than4 workers. Start serial; increase only after stable completions. Initialization serialized. Small pools stay resident; large pools offload BF16 models sequentially.

Atomic fsync JSON metadata after every image; PNG rename followed by SHA verification. Worker heartbeat15sec, controller status/reports60sec. OOM reduces concurrency and retries with backoff, maximum3; code errors are blocked, never patched by the daemon. Stale heartbeat/progress triggers owned-process SIGTERM then bounded SIGKILL. Valid finished images are skipped. Native validation checkpoints each response. Commercial incomplete traces stop for inspection instead of blindly issuing duplicate scored requests.

Monitoring is local and consumes no language-model tokens: `status.json`, `STATUS.md`, `logs/`, `scheduler.json`, `attempts/`, and generated report tables. A connected agent is still required to repair genuine code defects. Quota/auth/service failures may prevent Stage2 completion.

## Verification and deliverables

Tests cover family/native-layer legality, fixed loss independence from unrelated classes, RNG partition invariance, fsync state/checksum corruption, resource admission, simulated OOM/backoff, code-error blocking, orphan receipt reconciliation, SIGTERM/SIGKILL, file lock/reacquisition, fake50-image CLI parsing and no duplicate scored requests. Real CUDA calibration/smoke and controlled controller-kill recovery are recorded under `preflight/`; inspect actual receipts before claiming passed.

Outputs: per-condition/per-image/per-direction CSV; BF16 events; actual visual-weight fingerprints; gradients/cancellation/directed cross-objective gains; representation shift/variance; CKA/RSA; ASR–time/VRAM plots; native model–layer heatmap; frozen method lock; GPT raw request evidence and summary; `reports/PAPER_HANDOFF.md`. Many figures appear only once relevant valid results exist, never as invented values.

Remaining limitations: no machine-reboot auto-service installed, no automatic algorithm repair, no claim of exhaustive historical external-image audit, no proof that lowCKA means useful complementarity, no statistical-equivalence claim from5pp tolerance, and no faithful external baseline results until separately executed.
