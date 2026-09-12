#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$R2_ROOT/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: expected repository Python at $PYTHON_BIN; install/use the frozen project environment" >&2
  exit 2
fi
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
for group in separation rank localization distribution performance_bridge r2_r3_linkage final; do
  rm -rf "$R2_ROOT/cache/$group"
done
mkdir -p "$R2_ROOT/cache"
LOG="$R2_ROOT/cache/final/run.log"
mkdir -p "$(dirname "$LOG")"
{
  echo "R2 pipeline start $(date -u +%FT%TZ)"
  "$PYTHON_BIN" "$SCRIPT_DIR/01_parameter_separation_icself.py"
  "$PYTHON_BIN" "$SCRIPT_DIR/02_rank_reorganization.py"
  "$PYTHON_BIN" "$SCRIPT_DIR/03_coordinate_localization_icself.py"
  "$PYTHON_BIN" "$SCRIPT_DIR/04_distribution_reference_sensitivity.py"
  "$PYTHON_BIN" "$SCRIPT_DIR/05_outlet_parameter_bridge.py"
  "$PYTHON_BIN" "$SCRIPT_DIR/06_r2_r3_rank_linkage_audit.py"
  "$PYTHON_BIN" "$SCRIPT_DIR/99_validate_frozen_results.py"
  echo "R2 pipeline complete $(date -u +%FT%TZ)"
} 2>&1 | tee "$LOG"
(
  cd "$R2_ROOT/cache"
  find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256
)
echo "R2 ANALYSIS PIPELINE PASS"
