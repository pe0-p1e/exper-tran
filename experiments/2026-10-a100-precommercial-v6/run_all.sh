#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/outputs/a100_precommercial_v6"
LOGS="$OUT/logs"
PY="$REPO_ROOT/.venv-a100/bin/python"
mkdir -p "$LOGS" "$OUT/audits/stages"

if [[ ! -x "$PY" ]]; then
  echo "Missing prepared venv: $PY" >&2
  exit 2
fi
source "$REPO_ROOT/.venv-a100/bin/activate"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf-cache}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-$REPO_ROOT/data/imagenet_diverse10_minimal}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec 9>"$OUT/run_all.lock"
if ! flock -n 9; then
  echo "Another V6 run_all.sh process holds $OUT/run_all.lock" >&2
  exit 3
fi

exec > >(tee -a "$LOGS/run_all.log") 2>&1
echo "[run_all] start $(date -u +%FT%TZ) branch=$(git branch --show-current) torch=$($PY -c 'import torch; print(torch.__version__)')"

run_stage() {
  local stage="$1"
  local marker="$OUT/audits/stages/${stage}.complete"
  local log="$LOGS/${stage}.log"
  if [[ -f "$marker" ]]; then
    echo "[run_all] resume: $stage already complete"
    return 0
  fi
  echo "[run_all] >>> $stage $(date -u +%FT%TZ)"
  set +e
  "$PY" "$REPO_ROOT/experiments/2026-10-a100-precommercial-v6/src/campaign.py" "$stage" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -eq 0 ]]; then
    printf '%s\n' "$(date -u +%FT%TZ)" > "$marker"
    echo "[run_all] <<< $stage complete"
  else
    echo "[run_all] !!! $stage failed rc=$rc; stopping at global stage boundary"
    return "$rc"
  fi
}

for stage in \
  00_preflight \
  01_model_audit \
  02_dataset_prepare \
  03_clean_screen \
  04_batch_calibration \
  smoke \
  05_single_proxy \
  06_embedding_layers \
  07_multiple_proxy \
  08_ablation \
  09_reverse_direction \
  10_embedding_extraction \
  11_representation_analysis \
  12_joint_pca_tsne \
  13_asymmetry \
  14_correlation \
  15_validation \
  16_final_report; do
  run_stage "$stage"
done

echo "[run_all] finished $(date -u +%FT%TZ)"
