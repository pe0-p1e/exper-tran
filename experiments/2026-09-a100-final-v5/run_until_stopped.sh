#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WAIT_FOR_PID=""
BATCH_SIZE=50
while (($#)); do
  case "$1" in
    --wait-for-pid)
      WAIT_FOR_PID="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$ROOT/.hf-cache}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-$ROOT/data/imagenet_diverse10_minimal}"
export PYTHONPATH="$ROOT/src:$ROOT/experiments/2026-09-a100-final-v5/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

OUTPUT="$ROOT/outputs/a100_final_v5"
LOG_DIR="$OUTPUT/logs/queue"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/run_until_stopped_$STAMP.log"
GPU_LOG="$LOG_DIR/gpu_usage_continuous_$STAMP.csv"
PID_FILE="$LOG_DIR/run_until_stopped.pid"
printf '%s\n' "$$" >"$PID_FILE"

if [[ -n "$WAIT_FOR_PID" ]]; then
  echo "waiting_for_previous_queue_pid=$WAIT_FOR_PID start=$(date -u +%FT%TZ)" | tee -a "$LOG"
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 30; done
  # GNU timeout may leave a detached worker finishing its current cell. Do not
  # start a second GPU queue until every process from the old runner is gone.
  while pgrep -f '[r]un_pair_wave.py|[r]un_a100_final.py' >/dev/null; do
    sleep 30
  done
  echo "previous_queue_finished=$(date -u +%FT%TZ)" | tee -a "$LOG"
fi

export A100_CONTINUOUS_LOG="$LOG"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 10 >"$GPU_LOG" 2>&1 &
MONITOR_PID=$!
WORKER_PID=""
stop_workers() {
  for old_pid in $(pgrep -f '[r]un_8h_experiment.sh|[t]imeout --signal=INT --kill-after=60s 8h' || true); do
    kill -INT "$old_pid" 2>/dev/null || true
  done
  if [[ -n "$WAIT_FOR_PID" ]]; then
    kill -TERM "$WAIT_FOR_PID" 2>/dev/null || true
  fi
  for child_pid in $(pgrep -f '[r]un_pair_wave.py|[r]un_a100_final.py' || true); do
    kill -INT "$child_pid" 2>/dev/null || true
  done
  if [[ -n "$WORKER_PID" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    kill -INT "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  kill "$MONITOR_PID" 2>/dev/null || true
}
trap stop_workers EXIT INT TERM

echo "continuous_queue_start=$(date -u +%FT%TZ) batch_size=$BATCH_SIZE" | tee -a "$LOG"
bash "$ROOT/experiments/2026-09-a100-final-v5/run_8h_experiment.sh" \
  --worker "$BATCH_SIZE" > >(tee -a "$LOG") 2>&1 &
WORKER_PID=$!
wait "$WORKER_PID"
echo "continuous_queue_done=$(date -u +%FT%TZ)" | tee -a "$LOG"
