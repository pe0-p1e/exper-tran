#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
MAIN_REPO="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
export V7_DATA_ROOT="${V7_DATA_ROOT:-$MAIN_REPO/outputs/a100_precommercial_v7_targeted50}"
mkdir -p "$V7_DATA_ROOT/proxy_composition_18h/commercial_cli_campaign/logs"
exec > >(tee -a "$V7_DATA_ROOT/proxy_composition_18h/commercial_cli_campaign/logs/run_all.log") 2>&1
exec "$MAIN_REPO/.venv-a100/bin/python" -u "$REPO/experiments/2026-10-a100-precommercial-v7-targeted-50/src/run_commercial_cli.py" "$@"
