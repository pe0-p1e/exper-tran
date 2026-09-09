#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/experiments/2026-08-pull-push-multiclass-v4/src"
PYTHON="$ROOT/.venv-primary-ml-cka/bin/python"
exec 9>/tmp/exper-tran-linear8.lock
flock -n 9 || { echo 'Linear8 comparison already running'; exit 1; }
"$PYTHON" -m pytest -q experiments/2026-08-pull-push-multiclass-v4/tests tests/unit/attack/test_semantic_contrastive.py
"$PYTHON" -u experiments/2026-08-pull-push-multiclass-v4/src/run_linear_comparison.py "$@"
