#!/usr/bin/env bash
# Durable V7 campaign; artifacts live on persistent root disk, weights on ephemeral.
set -Eeuo pipefail
trap 'echo "V7 interrupted; completed batches remain resumable"; exit 130' INT TERM
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/2026-10-a100-precommercial-v7-targeted-50"
OUT="$ROOT/outputs/a100_precommercial_v7_targeted50"
PY="$ROOT/.venv-a100/bin/python"
export HF_HOME="${HF_HOME:-/ephemeral/a100_precommercial_v6/model-cache/hf-cache}"
export PYTHONPATH="$ROOT/src:$EXP/src:$ROOT/experiments/2026-10-a100-precommercial-v6/src:$ROOT/experiments/2026-09-a100-final-v5/src${PYTHONPATH:+:$PYTHONPATH}"
export PRIMARY_ML_CKA_STRICT_BF16=1
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
MODE=focus
for arg in "$@"; do
  case "$arg" in
    --preflight) MODE=preflight;; --smoke) MODE=smoke;; --pilot|--focus) MODE=focus;; --legacy-full) MODE=full;; --full|--resume) :;;
    *) echo "Unknown option: $arg"; exit 2;;
  esac
done
if [[ "$MODE" == focus ]]; then exec bash "$EXP/run_focus_v7.sh"; fi
mkdir -p "$OUT/logs" "$OUT/audits" "$HF_HOME"
exec 9>"$OUT/campaign.lock"
flock -n 9 || { echo 'Another V7 launcher is active'; exit 2; }
exec > >(tee -a "$OUT/logs/run_all_v7.log") 2>&1
cd "$ROOT"
[[ "$(git branch --show-current)" == experiment/a100-precommercial-v7-targeted-50 ]] || { echo 'Wrong V7 branch'; exit 2; }
[[ -x "$PY" ]] || { echo 'Missing A100 Python environment'; exit 2; }
echo "[$(date -Is)] V7 mode=$MODE commit=$(git rev-parse HEAD)"
# Repeat only unfinished artifacts after transient failures. A genuine external
# blocker remains visible in the final nonzero exit and failure audit.
ATTEMPTS="${V7_RETRY_ATTEMPTS:-3}"
for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
  echo "[$(date -Is)] campaign attempt $attempt/$ATTEMPTS"
  if "$PY" -u "$EXP/src/campaign_v7.py" --mode "$MODE" --resume; then
    echo "[$(date -Is)] requested V7 mode finished"
    exit 0
  fi
  if ((attempt<ATTEMPTS)); then
    echo 'Retry in 60 seconds; validated artifacts will be reused.'
    sleep 60
  fi
done
echo "V7 incomplete. See $OUT/audits/campaign_failures.json and logs."
exit 1
