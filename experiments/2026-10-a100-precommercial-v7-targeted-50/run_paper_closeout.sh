#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
MAIN="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
export V8_MAIN_REPO="$MAIN" V8_OUTPUT_ROOT="$MAIN/outputs/a100_precommercial_v8_joint_selection24h"
export V7_DATA_ROOT="$MAIN/outputs/a100_precommercial_v7_targeted50"
export HF_HOME="$MAIN/.hf-cache" PYTHONPATH="$MAIN/src" PRIMARY_ML_CKA_STRICT_BF16=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mode="${1:---status}"
shift || true
case "$mode" in
 --start|--resume) action=start ;;
 --status) action=status ;;
 --report) action=report ;;
 *) echo 'Use --start [--deadline-hours 24], --status or --report'; exit 2 ;;
esac
exec "$MAIN/.venv-a100/bin/python" -u "$REPO/experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/paper_closeout/launch.py" "$action" "$@"
