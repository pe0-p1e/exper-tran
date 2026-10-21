#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN_REPO="$(dirname "$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)")"
export V8_MAIN_REPO="$MAIN_REPO" PYTHONPATH="$MAIN_REPO/src" HF_HOME="$MAIN_REPO/.hf-cache"
exec "$MAIN_REPO/.venv-a100/bin/python" "$REPO/experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/stage1_methods/planner.py" "${1:-prepare}"
