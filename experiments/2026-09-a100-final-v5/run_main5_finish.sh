#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$ROOT/.hf-cache}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-$ROOT/data/imagenet_diverse10_minimal}"
export PYTHONPATH="$ROOT/src:$ROOT/experiments/2026-09-a100-final-v5/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

PYTHON="$ROOT/.venv-a100/bin/python"
EXPERIMENT="$ROOT/experiments/2026-09-a100-final-v5"
CONFIG="$EXPERIMENT/config/full90_selected_layer_main5.yaml"
OUTPUT="$ROOT/outputs/a100_final_v5"
LOG_DIR="$OUTPUT/logs/main5_finish"
mkdir -p "$LOG_DIR"
printf '%s\n' "$$" >"$LOG_DIR/runner.pid"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/main5_finish_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

PAIRS=(P02 P14 P16 P19 P23)
echo "resume_small_to_large_attack pair=P19 start=$(date -u +%FT%TZ)"
"$PYTHON" "$EXPERIMENT/src/run_transition_shards.py" \
  --config "$CONFIG" --output-dir "$OUTPUT" --pair P19 \
  --max-workers 4 --gpu-memory-budget-gb 70 --memory-reserve-gb 6

echo "main5_evaluation_start=$(date -u +%FT%TZ)"
PRE_EVAL_PID_FILE="$LOG_DIR/pre_evaluation.pid"
if [[ -f "$PRE_EVAL_PID_FILE" ]]; then
  PRE_EVAL_PID="$(cat "$PRE_EVAL_PID_FILE")"
  while kill -0 "$PRE_EVAL_PID" 2>/dev/null; do
    echo "waiting_for_concurrent_evaluation pid=$PRE_EVAL_PID time=$(date -u +%FT%TZ)"
    sleep 30
  done
fi
"$PYTHON" "$EXPERIMENT/src/run_pair_wave.py" \
  --config "$CONFIG" --output-dir "$OUTPUT" --stage evaluate \
  --pairs "${PAIRS[@]}" --max-batch-size 50 --max-workers 4 \
  --gpu-memory-budget-gb 70 --memory-reserve-gb 6

echo "split_summary_start=$(date -u +%FT%TZ)"
"$PYTHON" "$EXPERIMENT/src/split_full90_summary.py" \
  --config "$CONFIG" \
  --summary "$OUTPUT/summaries/single_proxy_full90_main5.csv" \
  --output-dir "$OUTPUT/summaries"

for pair in "${PAIRS[@]}"; do
  echo "geometry_analysis_start=$(date -u +%FT%TZ) pair=$pair"
  "$PYTHON" "$EXPERIMENT/src/analyze_joint_geometry.py" \
    --config "$CONFIG" --output-dir "$OUTPUT" --pair "$pair" \
    --batch-size 50 --plot-transition T0102 \
    --analysis-subdir "single_proxy_full90_main5/$pair" --resume
done
echo "package_results_start=$(date -u +%FT%TZ)"
"$PYTHON" "$EXPERIMENT/src/package_main_results.py" \
  --repo "$ROOT" --output outputs/a100_final_v5/shareable
echo "main5_finish_done=$(date -u +%FT%TZ)"
