#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
MAIN_REPO="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
export V8_MAIN_REPO="$MAIN_REPO"
export V7_DATA_ROOT="$MAIN_REPO/outputs/a100_precommercial_v7_targeted50"
export V8_OUTPUT_ROOT="${V8_OUTPUT_ROOT:-$MAIN_REPO/outputs/a100_precommercial_v8_joint_selection24h}"
export PYTHONPATH="$MAIN_REPO/src"
export HF_HOME="$MAIN_REPO/.hf-cache"
export PRIMARY_ML_CKA_STRICT_BF16=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PACKAGE="$REPO/experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h"
if [[ "${1:-}" == "--internal-supervisor" ]]; then
 shift
 exec "$MAIN_REPO/.venv-a100/bin/python" -u "$PACKAGE/watchdog.py" "$@"
fi
exec "$MAIN_REPO/.venv-a100/bin/python" -u "$PACKAGE/cli.py" "$@"
