#!/usr/bin/env bash
set -euo pipefail

SOURCE="/ephemeral/a100_final_v5/outputs/a100_final_v5/"
DEST="/home/ubuntu/experiment_backups/a100_final_v5/"
LOG_DIR="/home/ubuntu/experiment_backups/logs"
mkdir -p "$DEST" "$LOG_DIR"
exec 9>"$LOG_DIR/sync_persistent_results.lock"
flock -n 9 || exit 0

while true; do
  timestamp="$(date -u +%FT%TZ)"
  printf 'sync_start=%s\n' "$timestamp" >> "$LOG_DIR/sync_persistent_results.log"
  if rsync -a --partial --stats "$SOURCE" "$DEST" >> "$LOG_DIR/sync_persistent_results.log" 2>&1; then
    date -u +%FT%TZ > "$LOG_DIR/last_successful_sync_utc.txt"
    printf 'sync_success=%s\n' "$(date -u +%FT%TZ)" >> "$LOG_DIR/sync_persistent_results.log"
  else
    printf 'sync_failed=%s\n' "$(date -u +%FT%TZ)" >> "$LOG_DIR/sync_persistent_results.log"
  fi
  sleep 120
done
