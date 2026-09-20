#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="src:experiments/2026-08-pull-push-multiclass-v4/src"
export IMAGENET_ROOT="${IMAGENET_ROOT:-$PWD/data/imagenet_diverse10_minimal}"
PYTHON=".venv-primary-ml-cka/bin/python"
CONFIG8="experiments/2026-09-threeclass-transfer/config/threeclass.yaml"
ARMS8="experiments/2026-09-threeclass-transfer/config/p14_p19_tuning_arms.yaml"
OUT8="outputs/threeclass_transfer_8"
CONFIG50_TEMPLATE="experiments/2026-09-threeclass-transfer/config/threeclass_scale50.yaml"
CONFIG50="experiments/2026-09-threeclass-transfer/config/threeclass_scale50_selected.yaml"
OUT50="outputs/threeclass_transfer_50"

# Search P14 and P19 on the already-screened 8-image cohorts.
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$CONFIG8" --arm-config "$ARMS8" --output-dir "$OUT8" \
  --pairs P14 P19 --transitions G01 G02 G03 G04 G05 G06 \
  --arms p14_p19_full_target_pull p14_p19_full_push_strong \
         p14_p19_second_target_pull p14_p19_second_target_pull_half \
  --resume --fail-on-error

"$PYTHON" experiments/2026-09-threeclass-transfer/src/select_best_scale50.py \
  --states "$OUT8/states" --template "$CONFIG50_TEMPLATE" --output "$CONFIG50"

# Rebuild a 50-image common-clean cohort, then attack all six directions using
# the selected P14/P19 arms and the previously selected P16 arm.
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/prepare_data.py \
  --config "$CONFIG50" --output-dir "$OUT50"
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/screen_transitions.py \
  --config "$CONFIG50" --output-dir "$OUT50" --resume
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_scale50.py \
  --config "$CONFIG50" --output-dir "$OUT50" \
  --pairs P14 P16 P19 --transitions G01 G02 G03 G04 G05 G06 --resume
