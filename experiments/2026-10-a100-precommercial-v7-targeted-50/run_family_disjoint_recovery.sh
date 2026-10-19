#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-/ephemeral/a100_precommercial_v6/model-cache/hf-cache}"
export PRIMARY_ML_CKA_STRICT_BF16=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/outputs/a100_precommercial_v7_targeted50/family_disjoint_recovery"
mkdir -p "$OUT/logs"
exec 9>"$ROOT/outputs/a100_precommercial_v7_targeted50/campaign.lock"
flock -n 9 || { echo 'Another V7 campaign is active'; exit 2; }
exec > >(tee -a "$OUT/logs/run_all.log") 2>&1
PY="$ROOT/.venv-a100/bin/python"
"$PY" -m pytest -q experiments/2026-10-a100-precommercial-v7-targeted-50/tests
exec "$PY" -u experiments/2026-10-a100-precommercial-v7-targeted-50/src/run_family_disjoint.py
