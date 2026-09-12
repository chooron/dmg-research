#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$R1_ROOT/../../../../" && pwd)"
PYTHON="$REPO/.venv/bin/python"
export PYTHONPATH="$R1_ROOT/scripts:$REPO/project/benchmark:$REPO"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" "$R1_ROOT/scripts/00_audit_inputs.py"
"$PYTHON" "$R1_ROOT/scripts/01_build_fulltest_pair_table.py"
"$PYTHON" "$R1_ROOT/scripts/02_temporal_ab_forward.py"
"$PYTHON" "$R1_ROOT/scripts/03_summarize_r1.py"
"$PYTHON" "$R1_ROOT/scripts/08_run_sensitivity.py"
"$PYTHON" "$R1_ROOT/scripts/04_plot_fig1a.py"
"$PYTHON" "$R1_ROOT/scripts/05_plot_fig1b.py"
"$PYTHON" "$R1_ROOT/scripts/06_plot_fig1c.py"
"$PYTHON" "$R1_ROOT/scripts/07_plot_fig1_combined.py"
"$PYTHON" "$R1_ROOT/scripts/09_write_report.py"

echo "PASS: complete R1 pipeline finished; outputs are under $R1_ROOT"
