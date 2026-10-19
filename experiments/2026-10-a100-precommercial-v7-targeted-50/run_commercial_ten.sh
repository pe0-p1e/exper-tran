#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
MAIN_REPO="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
export V7_DATA_ROOT="${V7_DATA_ROOT:-$MAIN_REPO/outputs/a100_precommercial_v7_targeted50}"
DEST="$V7_DATA_ROOT/proxy_composition_18h/commercial_ten_image"
mkdir -p "$DEST/logs"
exec 9>"$DEST/campaign.lock"
flock -n 9
exec > >(tee -a "$DEST/logs/run_all.log") 2>&1
PY="$MAIN_REPO/.venv-a100/bin/python"
SRC="$REPO/experiments/2026-10-a100-precommercial-v7-targeted-50/src"
"$PY" "$SRC/prepare_commercial_ten.py" --reuse
if [[ "${1:-}" == "--prepare-only" ]]; then exit 0; fi
"$PY" -u "$SRC/run_codex_cli_pilot.py" --model gpt-6.1-sol --images 10 \
  --pack-dir "$V7_DATA_ROOT/commercial_pilot10" \
  --result-dir "$DEST/codex/gpt-6.1-sol"
"$PY" "$SRC/report_commercial_ten.py"
