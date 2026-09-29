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
CONFIG="$ROOT/experiments/2026-09-a100-final-v5/config/a100_final.yaml"
OUTPUT="$ROOT/outputs/a100_final_v5"
BATCH_SIZE="${1:-50}"
LOG_DIR="$OUTPUT/logs/queue"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/run_8h_$STAMP.log"
GPU_LOG="$LOG_DIR/gpu_usage_$STAMP.csv"

if [[ "${1:-}" == "--worker" ]]; then
  BATCH_SIZE="${2:-50}"
  printf 'queue_start=%s batch_size=%s\n' "$(date -u +%FT%TZ)" "$BATCH_SIZE"
  "$PYTHON" experiments/2026-09-a100-final-v5/src/prepare_layer_configs.py --config "$CONFIG"
  for pair in P02 P14 P16 P19 P21 P22; do
    CALIBRATION_REPORT="$OUTPUT/calibration/${pair}_T01_batch_sizes.csv"
    if [[ -s "$CALIBRATION_REPORT" ]] && "$PYTHON" - "$CALIBRATION_REPORT" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
valid = [
    row for row in rows if row["status"] == "ok" and float(row["peak_reserved_gb"]) <= 75
]
raise SystemExit(0 if valid else 1)
PY
    then
      printf 'calibration_reuse=%s pair=%s\n' "$(date -u +%FT%TZ)" "$pair"
    else
      printf 'calibration_start=%s pair=%s\n' "$(date -u +%FT%TZ)" "$pair"
      "$PYTHON" experiments/2026-09-a100-final-v5/src/calibrate_a100_batch.py \
        --config "$CONFIG" --output-dir "$OUTPUT" --pair "$pair" --transition T01 \
        --candidates 8 16 24 32 48 50 --memory-limit-gb 75
    fi
  done
  PAIRS=(P02 P14 P16 P19 P20 P21 P22 P23)
  calibrated_batch() {
    case "$1" in
      P02|P20) calibration_pair=P02 ;;
      P14|P23) calibration_pair=P14 ;;
      P16) calibration_pair=P16 ;;
      P19) calibration_pair=P19 ;;
      P21) calibration_pair=P21 ;;
      P22) calibration_pair=P22 ;;
    esac
    "$PYTHON" - "$OUTPUT/calibration/${calibration_pair}_T01_batch_sizes.csv" "$BATCH_SIZE" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
valid = [
    int(row["batch_size"])
    for row in rows
    if row["status"] == "ok" and float(row["peak_reserved_gb"]) <= 75
]
if not valid:
    raise SystemExit(f"No safe batch size in {sys.argv[1]}")
print(min(max(valid), int(sys.argv[2])))
PY
  }
  for percent in 001 025 050 075 100; do
    LAYER_CONFIG="$ROOT/experiments/2026-09-a100-final-v5/config/layers/layer_${percent}.yaml"
    printf 'attack_wave_start=%s layer_percent=%s\n' "$(date -u +%FT%TZ)" "$percent"
    "$PYTHON" experiments/2026-09-a100-final-v5/src/run_pair_wave.py \
      --config "$LAYER_CONFIG" --output-dir "$OUTPUT" --stage attack \
      --pairs "${PAIRS[@]}" --max-batch-size "$BATCH_SIZE" --max-workers 4 \
      --gpu-memory-budget-gb 70 --memory-reserve-gb 6
    printf 'evaluation_wave_start=%s layer_percent=%s\n' "$(date -u +%FT%TZ)" "$percent"
    "$PYTHON" experiments/2026-09-a100-final-v5/src/run_pair_wave.py \
      --config "$LAYER_CONFIG" --output-dir "$OUTPUT" --stage evaluate \
      --pairs "${PAIRS[@]}" --max-batch-size "$BATCH_SIZE" --max-workers 4 \
      --gpu-memory-budget-gb 70 --memory-reserve-gb 6
    for pair in "${PAIRS[@]}"; do
      pair_batch="$(calibrated_batch "$pair")"
      printf 'analysis_start=%s layer_percent=%s pair=%s batch_size=%s\n' \
        "$(date -u +%FT%TZ)" "$percent" "$pair" "$pair_batch"
      "$PYTHON" experiments/2026-09-a100-final-v5/src/analyze_joint_geometry.py \
        --config "$LAYER_CONFIG" --output-dir "$OUTPUT" --pair "$pair" \
        --batch-size "$pair_batch" --plot-transition T01 \
        --analysis-subdir "layers/layer_${percent}/${pair}" --resume
    done
  done
  # Controlled loss ablation follows the complete single-proxy/layer sweep.
  # Keep the same images, seed, steps, proxy-target pairings, and common
  # optimizer settings; only the pull/push terms change between arms.
  for arm in pull_only push_only pull_push; do
    ABLATION_CONFIG="$ROOT/experiments/2026-09-a100-final-v5/config/ablation_${arm}.yaml"
    "$PYTHON" experiments/2026-09-a100-final-v5/src/prepare_ablation_config.py \
      --config "$CONFIG" --output "$ABLATION_CONFIG" --arm "$arm"
    printf 'ablation_start=%s arm=%s\n' "$(date -u +%FT%TZ)" "$arm"
    "$PYTHON" experiments/2026-09-a100-final-v5/src/run_pair_wave.py \
      --config "$ABLATION_CONFIG" --output-dir "$OUTPUT" --stage attack \
      --pairs P23 P02 P14 --max-batch-size "$BATCH_SIZE" --max-workers 4 \
      --gpu-memory-budget-gb 70 --memory-reserve-gb 6
    "$PYTHON" experiments/2026-09-a100-final-v5/src/run_pair_wave.py \
      --config "$ABLATION_CONFIG" --output-dir "$OUTPUT" --stage evaluate \
      --pairs P23 P02 P14 --max-batch-size "$BATCH_SIZE" --max-workers 4 \
      --gpu-memory-budget-gb 70 --memory-reserve-gb 6
  done
  printf 'analysis_start=%s\n' "$(date -u +%FT%TZ)"
  "$PYTHON" experiments/2026-09-a100-final-v5/src/audit_model_layers.py \
    --config "$CONFIG" --output "$OUTPUT/analysis/model_layer_audit.csv"
  printf 'queue_done=%s\n' "$(date -u +%FT%TZ)"
  exit 0
fi

 nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 10 >"$GPU_LOG" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
timeout --signal=INT --kill-after=60s 8h bash "$0" --worker "$BATCH_SIZE" 2>&1 | tee -a "$LOG"
