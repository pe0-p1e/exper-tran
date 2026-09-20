#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="src:experiments/2026-08-pull-push-multiclass-v4/src"
PYTHON=".venv-primary-ml-cka/bin/python"
CONFIG="experiments/2026-09-threeclass-transfer/config/threeclass.yaml"
OUT="outputs/threeclass_transfer_8"
if [[ -z "${IMAGENET_ROOT:-}" && -d "data/imagenet_diverse10_minimal/train" ]]; then
  export IMAGENET_ROOT="$PWD/data/imagenet_diverse10_minimal"
fi

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/prepare_data.py \
  --config "$CONFIG" --output-dir "$OUT"
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/screen_transitions.py \
  --config "$CONFIG" --output-dir "$OUT" --resume
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$CONFIG" --output-dir "$OUT" \
  --pairs P14 P16 P19 \
  --transitions G01 G02 G03 G04 G05 G06 \
  --arms full_pull_push second_loss_only \
  --resume --fail-on-error
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/summarize.py \
  --output-dir "$OUT"
