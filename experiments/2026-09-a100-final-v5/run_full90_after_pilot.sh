#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WAIT_PID="${1:?pass the current run_until_stopped PID}"
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
OUTPUT="$ROOT/outputs/a100_final_v5"
BASE_CONFIG="$EXPERIMENT/config/a100_final.yaml"
FULL_CONFIG="$EXPERIMENT/config/full90_selected_layer.yaml"
LOG_DIR="$OUTPUT/logs/full90"
mkdir -p "$LOG_DIR"
printf '%s\n' "$$" >"$LOG_DIR/runner.pid"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/full90_$STAMP.log"

echo "waiting_for_pilot_pid=$WAIT_PID start=$(date -u +%FT%TZ)" | tee -a "$LOG"
if [[ "$WAIT_PID" != "0" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi
while pgrep -f '[r]un_pair_wave.py|[r]un_a100_final.py' >/dev/null; do sleep 20; done
if [[ -e "$LOG_DIR/STOP" ]]; then
  echo "stop_marker_present=$(date -u +%FT%TZ); leaving follow-up queue idle" | tee -a "$LOG"
  exit 0
fi

echo "pilot_done=$(date -u +%FT%TZ); selecting layer" | tee -a "$LOG"
"$PYTHON" "$EXPERIMENT/src/select_pilot_layer.py" \
  --config "$BASE_CONFIG" \
  --summary-dir "$OUTPUT/summaries" \
  --output-config "$FULL_CONFIG" \
  --selection-report "$OUTPUT/analysis/pilot_layer_selection.json"

echo "prepare_full90=$(date -u +%FT%TZ)" | tee -a "$LOG"
"$PYTHON" "$EXPERIMENT/src/prepare_data.py" --config "$FULL_CONFIG" --output-dir "$OUTPUT"
echo "screen_full90=$(date -u +%FT%TZ)" | tee -a "$LOG"
"$PYTHON" "$EXPERIMENT/src/screen_transitions.py" \
  --config "$FULL_CONFIG" --output-dir "$OUTPUT" --resume

PAIRS=(P02 P14 P16 P19 P20 P21 P22 P23)
echo "attack_full90_start=$(date -u +%FT%TZ)" | tee -a "$LOG"
"$PYTHON" "$EXPERIMENT/src/run_pair_wave.py" \
  --config "$FULL_CONFIG" --output-dir "$OUTPUT" --stage attack \
  --pairs "${PAIRS[@]}" --max-batch-size 50 --max-workers 4 \
  --gpu-memory-budget-gb 70 --memory-reserve-gb 6
echo "evaluate_full90_start=$(date -u +%FT%TZ)" | tee -a "$LOG"
"$PYTHON" "$EXPERIMENT/src/run_pair_wave.py" \
  --config "$FULL_CONFIG" --output-dir "$OUTPUT" --stage evaluate \
  --pairs "${PAIRS[@]}" --max-batch-size 50 --max-workers 4 \
  --gpu-memory-budget-gb 70 --memory-reserve-gb 6
"$PYTHON" "$EXPERIMENT/src/split_full90_summary.py" \
  --config "$FULL_CONFIG" \
  --summary "$OUTPUT/summaries/single_proxy_full90.csv" \
  --output-dir "$OUTPUT/summaries"

for pair in "${PAIRS[@]}"; do
  echo "analysis_full90_start=$(date -u +%FT%TZ) pair=$pair" | tee -a "$LOG"
  "$PYTHON" "$EXPERIMENT/src/analyze_joint_geometry.py" \
    --config "$FULL_CONFIG" --output-dir "$OUTPUT" --pair "$pair" \
    --batch-size 50 --plot-transition T0102 \
    --analysis-subdir "single_proxy_full90/$pair" --resume
done
echo "full90_done=$(date -u +%FT%TZ)" | tee -a "$LOG"
