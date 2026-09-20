#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="src:experiments/2026-08-pull-push-multiclass-v4/src"
export IMAGENET_ROOT="${IMAGENET_ROOT:-$PWD/data/imagenet_diverse10_minimal}"
PYTHON=".venv-primary-ml-cka/bin/python"
CONFIG="experiments/2026-09-threeclass-transfer/config/threeclass.yaml"
ARMS="experiments/2026-09-threeclass-transfer/config/p16_tuning_arms.yaml"
OUT="outputs/threeclass_transfer_8"

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$CONFIG" --arm-config "$ARMS" --output-dir "$OUT" \
  --pairs P16 --transitions G01 G02 G03 G04 G05 G06 \
  --arms p16_full_target_pull p16_full_push_strong \
         p16_second_target_pull p16_second_target_pull_half \
  --resume --fail-on-error
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/summarize.py \
  --output-dir "$OUT"
