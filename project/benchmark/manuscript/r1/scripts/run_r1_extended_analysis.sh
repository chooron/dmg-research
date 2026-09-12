#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$R1_ROOT/../../../../" && pwd)"
PYTHON="$REPO/.venv/bin/python"
export PYTHONPATH="$R1_ROOT/scripts:$REPO/project/benchmark:$REPO"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" "$R1_ROOT/scripts/10_model_estimator_tendency.py"
"$PYTHON" "$R1_ROOT/scripts/11_basin_estimator_tendency.py"
"$PYTHON" "$R1_ROOT/scripts/12_two_way_decomposition.py"
"$PYTHON" "$R1_ROOT/scripts/13_temporal_basin_persistence.py"
"$PYTHON" "$R1_ROOT/scripts/14_basin_difficulty_checks.py"
"$PYTHON" "$R1_ROOT/scripts/15_plot_r1_extended_diagnostics.py"
"$PYTHON" "$R1_ROOT/scripts/R1X_write_report.py"

echo "PASS: R1X extended analysis complete; formal Fig1 files were not modified"
