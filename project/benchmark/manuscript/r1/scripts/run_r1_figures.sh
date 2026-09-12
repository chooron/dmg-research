#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$R1_ROOT/../../../../" && pwd)"
PYTHON="$REPO/.venv/bin/python"
export PYTHONPATH="$R1_ROOT/scripts:$REPO/project/benchmark:$REPO"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# The final figure consumes existing R1/R1X outputs and prepares the
# derivative tables and Table S1; no training or forward run occurs.
"$PYTHON" "$R1_ROOT/scripts/20_prepare_fig1_data.py"
"$PYTHON" "$R1_ROOT/scripts/26_build_r1_supp_tables.py"
"$PYTHON" "$R1_ROOT/scripts/32_plot_fig1_final.py"
"$PYTHON" "$R1_ROOT/scripts/R1X_write_figure_build_report.py"

echo "PASS: revised PNG-only Figure 1 build complete"
