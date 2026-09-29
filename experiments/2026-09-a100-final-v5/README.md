# A100 final v5 runner

This experiment branch keeps the v4 dependency lock and adds an A100 runner with BF16 generative proxies, configurable attack batches, classification-free pull/push steps, one-step batch calibration, and joint target-space geometry analysis.

The controlled generative pair matrix is `P02`, `P14`, `P16`, `P19`, `P20`, `P21`, `P22`, and `P23`, spanning cross-family transfer and both scale directions within Qwen, InternVL, and Gemma. The 10 directed transitions are five reverse class pairs so the asymmetry table can compare both directions while covering all 10 classes. Attack settings are epsilon `16/255`, 50 steps, step size `1/255`, momentum 1, random start, seed 42, `lambda_cls=0`, and the linear objective `1.5 * (1-cos(target)) + 0.5 * (1+cos(source))`.

Run from the repository root after activating the pinned Python 3.11 environment and setting `CUDA_VISIBLE_DEVICES`, `HF_HOME`, `IMAGENET_ROOT`, `PYTHONPATH`, and the CUDA allocator options described in the experiment notes. Keep model snapshots local and do not update the v4 package pins during the formal run. The v5 scripts download only the exact model revisions named by the config and 112 ImageNet images per class:

```bash
export HF_HOME="$PWD/.hf-cache"
export IMAGENET_ROOT="$PWD/data/imagenet_diverse10_minimal"
export PYTHONPATH="$PWD/src:$PWD/experiments/2026-09-a100-final-v5/src"
python experiments/2026-09-a100-final-v5/src/download_models.py \
  --config experiments/2026-09-a100-final-v5/config/a100_final.yaml
python experiments/2026-09-a100-final-v5/src/download_required_imagenet.py \
  --config experiments/2026-09-a100-final-v5/config/a100_final.yaml
```

Prepare canonical references/candidates and common-clean cohorts in the v5 output tree:

```bash
export PYTHONPATH="$PWD/src:$PWD/experiments/2026-09-a100-final-v5/src"
python experiments/2026-09-a100-final-v5/src/prepare_data.py \
  --config experiments/2026-09-a100-final-v5/config/a100_final.yaml \
  --output-dir outputs/a100_final_v5
python experiments/2026-09-a100-final-v5/src/screen_transitions.py \
  --config experiments/2026-09-a100-final-v5/config/a100_final.yaml \
  --output-dir outputs/a100_final_v5
```

Calibrate each proxy on a real one-step attack at candidate batch sizes 8, 16, 24, 32, 48, and 50. The calibration uses a temporary output tree, records allocated/reserved peaks, and selects the largest candidate at or below 75 GiB:

```bash
python experiments/2026-09-a100-final-v5/src/calibrate_a100_batch.py \
  --config experiments/2026-09-a100-final-v5/config/a100_final.yaml \
  --output-dir outputs/a100_final_v5 --pair P23 --transition T01
```

Then run the exact smoke requested (two images, one transition, two attack steps, including frozen target generation):

```bash
python experiments/2026-09-a100-final-v5/src/run_a100_final.py \
  --config experiments/2026-09-a100-final-v5/config/a100_final.yaml \
  --output-dir outputs/a100_final_v5 --pair P23 --smoke --batch-size 2
```

After the smoke passes, the primary queue sweeps the auto-mapped vision embedding depths 1%, 25%, 50%, 75%, and 100% over all eight pair conditions and ten shared-clean directed transitions. At each layer, a calibrated, memory-budgeted worker pool runs proxy-only attacks in parallel, unloads all proxies, then runs frozen target-only PNG evaluations in parallel; layer geometry analysis follows serially. It resumes complete states and writes joint geometry diagnostics per layer and pair. When this finishes, it continues to the 50%-depth loss ablation for P23, P02, and P14, comparing pull-only, push-only, and pull+push with all other settings fixed:

```bash
experiments/2026-09-a100-final-v5/run_8h_experiment.sh 50
```

For an unbounded, resumable queue that keeps going through the ablation until
all queued work is complete, use `run_until_stopped.sh`. When switching from an
already-running bounded queue, pass its PID so the new queue waits for it to
exit and then resumes from saved states:

```bash
experiments/2026-09-a100-final-v5/run_until_stopped.sh --wait-for-pid 9988
```

The loss ablation uses the same fixed coefficient values as the main method:
pull-only `(1.5, 0)`, push-only `(0, 0.5)`, and combined `(1.5, 0.5)`. The combined
arm therefore has the same pull:push ratio of `3:1` as the primary experiment.

Add `--resume` to continue completed states when invoking the runner directly. Do not load the target while generating attacks; the runner unloads the proxy per attack batch and loads the target only for frozen PNG evaluation.

The runner stores state files under `states_a100_v5/`, PNGs under `attacks/`, and the aggregate table under `summaries/a100_final_results.csv`. Joint post-hoc outputs are written under `analysis/`: embeddings, per-class dispersion/covariance/effective rank, per-image and per-transition representation shifts, directed asymmetry, Spearman correlation CSV/heatmap, and joint PCA/t-SNE figures. PCA and t-SNE are each fitted once to the combined source-reference, target-reference, clean, and adversarial embedding set; t-SNE is qualitative only.
