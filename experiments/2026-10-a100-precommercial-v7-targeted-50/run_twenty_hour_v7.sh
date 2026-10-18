#!/usr/bin/env bash
# Compatibility keeps all completed ten-hour-study artifacts in their old paths.
set -Eeuo pipefail
export V7_RESEARCH_HOURS=20
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_ten_hour_v7.sh" "$@"
