#!/bin/bash
# Mem0 daily cleanup script — designed for systemd ExecStartPre integration
# 
# This script runs cleanup at most once per day using a date marker file.
# On success, the marker is updated with today's date. On failure, no marker
# update occurs so it will retry on next service start.

set -e

MARKER="/tmp/.mem0_cleanup_date"
TODAY=$(date +%Y-%m-%d)
CLEANED=$(cat "$MARKER" 2>/dev/null || echo "")

if [ "$TODAY" = "$CLEANED" ]; then
    # Already cleaned today — skip
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Mem0 lifecycle cleanup..."

# Run cleanup
if cd /path/to/mem0-deployment && python -m mem0_lifecycle.cleanup >> /tmp/mem0_cleanup.log 2>&1; then
    echo "$TODAY" > "$MARKER"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleanup completed successfully"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleanup failed — will retry on next start" >&2
    exit 1
fi
