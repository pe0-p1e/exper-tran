#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Focused V7 interrupted; resume preserves completed pilots"; exit 130' INT TERM
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-/ephemeral/a100_precommercial_v6/model-cache/hf-cache}"
export PRIMARY_ML_CKA_STRICT_BF16=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/outputs/a100_precommercial_v7_targeted50"
mkdir -p "$OUT/logs" "$OUT/focused_optimization"
exec 9>"$OUT/campaign.lock"
flock -n 9 || { echo 'Another V7 GPU campaign is active'; exit 2; }
exec > >(tee -a "$OUT/logs/run_focus_v7.log") 2>&1
"$ROOT/.venv-a100/bin/python" -m pytest -q experiments/2026-10-a100-precommercial-v7-targeted-50/tests
"$ROOT/.venv-a100/bin/python" -u experiments/2026-10-a100-precommercial-v7-targeted-50/src/focus_search.py
