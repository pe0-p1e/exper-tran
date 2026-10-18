# Ten-hour research campaign

The user approved the ten-hour research prompt on 2026-10-18. Default `run_all_v7.sh` now runs this campaign. Its direct entry point is:

```bash
bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_ten_hour_v7.sh
```

One tmux session owns the campaign lock. The budget is persisted in `outputs/a100_precommercial_v7_targeted50/ten_hour_research/budget.json`; resuming does not extend it. Valid completed work is reused. A watchdog enforces the remaining wall-time budget, and a finalizer saves available results, explicit missing conditions, validation, compact reports and an authorized Git upload. Completion before ten hours must be reported with actual runtime, never as ten hours of compute.

Read `ten_hour_plan.yaml` for frozen scientific settings and the candidate families. The algorithm ladder compares original Pull/Push, single-layer margin, soft-reference margin, four-layer margin, patch objectives and projector objectives. Small weighted mixtures precede proxy subset/composition and optimizer studies. Parameters are selected using native generated labels on independent surrogate models, never final targets. The initial pilot has five directions × two images; top candidates receive an explicitly nested eight-image-per-direction development confirmation. Nested confirmation is not an untouched generalization test.

Research ranking prioritizes the worst model-direction native ASR, then pooled ASR; runtime breaks ties. Existing results may be reused as labeled historical development context. Fair interpretation must account for different implementation fingerprints, cache effects and extra gradient calls. A nominally identical seed is not a substitute for multi-seed or implementation-matched reproducibility checks.

The selected ensemble, strongest validated single proxy and original Pull/Push on the selected ensemble are frozen before target evaluation. The real target stage uses the original five ×30 test images with an independent method lock. Additional development seed confirmations do not choose the seed. All main conditions keep strict BF16, no quantization, epsilon16/255, 50 steps and alpha1/255.

At most four GPU workers are permitted; the current conservative scheduler runs one GPU worker. It counts visual forwards, image forwards, backwards, runtime and reserved memory. It never quantizes to fit an ensemble. Models and unused BF16 language modules are offloaded as needed.

Results are under `outputs/a100_precommercial_v7_targeted50/ten_hour_research/`. Partial measurements are not zeros. Undefined legacy conditional TASR remains N/A when native proxy hit masks were not measured. The finalizer checks frozen PNG hashes, uint8 perturbation bounds and test-cohort identity, emits paired comparisons and exact binomial intervals, and optionally performs post-hoc target geometry if budget remains. It must not describe budget-skipped analysis as completed.

Full PNGs/checkpoints stay on persistent root storage. Only compact report data, audits and a few real examples are uploaded. Commercial APIs are not used.
