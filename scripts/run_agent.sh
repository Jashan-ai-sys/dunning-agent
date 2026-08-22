#!/usr/bin/env bash
# Keep the voice agent worker alive.
#
# The LiveKit worker gives up and exits after 16 failed reconnects, which a
# brief WiFi drop is enough to trigger. That is reasonable behaviour for a
# process under a supervisor -- this is the supervisor.
#
#   bash scripts/run_agent.sh
#
# Ctrl-C twice to stop (once kills the worker, twice kills the loop).

set -u
cd "$(dirname "$0")/.."

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "--- starting agent worker (run $attempt) at $(date '+%H:%M:%S') ---"
  PYTHONUNBUFFERED=1 uv run --group voice python -u -m app.voice.agent dev
  code=$?
  if [ $code -eq 130 ]; then
    echo "interrupted; stopping supervisor"
    exit 0
  fi
  echo "--- worker exited with $code, restarting in 3s ---"
  sleep 3
done
