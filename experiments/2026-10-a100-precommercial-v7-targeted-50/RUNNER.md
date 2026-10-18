# Long-running V7 runner

```bash
bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_all_v7.sh --full --resume
```

The default is the complete campaign. No wall-clock deadline, human approval or intermediate commands are required. Run inside tmux to survive terminal disconnection. VM hibernation stops computation; rerun the same command after resuming. Results/checkpoints are on the persistent root filesystem; model weights alone use ephemeral storage.

Stages:
1. Exact pinned model availability, strict BF16/CUDA, fixed ImageNet manifests, unit tests.
2. Original/new visual-extraction equivalence at five depths, nonzero finite image gradients and historical diagnostic export.
3. Actual four-layer gradient/memory calibration (2/4/8/16); use a conservative measured batch and sequential proxy gradient computation.
4. Two-image/two-step multi-proxy smokes for objective/fusion/DI variants plus batch-resume verification. These are labeled smoke, never final results.
5. Controlled objective ladder on 50 separate development images: L0; margin; soft-reference margin; four-layer margin; local/global patch; combined reference+multilayer; L0+L1; patch-only; individual 25/50/75% layers; two layers.
6. Disjoint Qwen4B/InternVL4B held-out-surrogate validation. Method selection ranks worst-direction representation hit rate, pooled representation hit rate, then margin. Representation hits are NOT MLLM output ASR.
7. With the independently selected loss: individual proxy/subset comparisons, equal/adaptive/PCGrad/adaptive+PCGrad/consensus fusion, DI, TI, and a separately labeled 3-restart budget.
8. Lock the multi-proxy finalist and all test conditions before final target inference. Resume from an existing lock never repeats algorithm selection.
9. Generate all locked 30-image × five-direction test conditions, matched standalone proxies, original equal baseline and robustness seeds 42/43/44.
10. Frozen PNG evaluation on Qwen27B, InternVL14B and Gemma31B; export exact successes/denominators, bootstrap comparisons, complete-cell/PNG/cohort/precision validation and report.

Unsuccessful conditions and downloads are logged. Independent cells continue. Up to three campaign-level retries handle transient failures, with completed artifact hashes verified before reuse. The script exits nonzero for remaining blockers; a successful process exit cannot substitute for complete-cell validation. Default main steps=50, epsilon=16/255, alpha=1/255, momentum=1, BF16 model weights throughout.

Useful outputs under `outputs/a100_precommercial_v7_targeted50/`:
- `logs/run_all_v7.log` and per-condition logs
- `audits/current_stage.json`, `performance_benchmark.json`, `method_lock.json`
- `attacks/dev`, `attacks/test` with batch states and exact PNG hashes
- `development_evaluations`, `evaluations` with image-level results
- `reports/diagnostics.md`, `target_direction_results.csv`, `paired_baselines.csv`, `final_report.md`

No new transfer result or 50% milestone is claimed by this implementation. Post-hoc target geometry and further follow-up benchmarks are separate measured deliverables; they cannot influence attack selection. Full feature analysis is being integrated and must be listed as pending until its outputs exist.
