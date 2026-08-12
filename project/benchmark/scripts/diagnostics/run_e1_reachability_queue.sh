#!/usr/bin/env bash
# Continue the resumable E1 CUDA matrix after an already-running worker exits.
set -euo pipefail

current_pid="$1"
while kill -0 "$current_pid" 2>/dev/null; do
  sleep 30
done

cd "$(dirname "$0")/../.."
exec env PYTHONPATH=. python scripts/diagnostics/e1_reachability.py --steps 500
