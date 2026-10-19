#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
MAIN_REPO="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
export V7_DATA_ROOT="${V7_DATA_ROOT:-$MAIN_REPO/outputs/a100_precommercial_v7_targeted50}"
export HF_HOME="${HF_HOME:-/ephemeral/a100_precommercial_v6/model-cache/hf-cache}"
export PRIMARY_ML_CKA_STRICT_BF16=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$V7_DATA_ROOT/proxy_composition_18h/logs"
exec > >(tee -a "$V7_DATA_ROOT/proxy_composition_18h/logs/run_all.log") 2>&1
exec "$MAIN_REPO/.venv-a100/bin/python" -u "$REPO/experiments/2026-10-a100-precommercial-v7-targeted-50/src/run_proxy_study.py" "$@"
