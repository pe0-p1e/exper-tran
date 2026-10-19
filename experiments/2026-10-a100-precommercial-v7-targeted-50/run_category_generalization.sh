#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-a100/bin/python"
if [[ " ${*} " == *' --prepare-only '* ]]; then exec "$PY" experiments/2026-10-a100-precommercial-v7-targeted-50/src/run_category_generalization.py "$@"; fi
export HF_HOME="${HF_HOME:-/ephemeral/a100_precommercial_v6/model-cache/hf-cache}"
export PRIMARY_ML_CKA_STRICT_BF16=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"$ROOT/outputs/a100_precommercial_v7_targeted50/campaign.lock"
flock -n 9 || { echo 'Another V7 campaign is active; category generalization must wait.'; exit 2; }
exec "$PY" -u experiments/2026-10-a100-precommercial-v7-targeted-50/src/run_category_generalization.py "$@"
