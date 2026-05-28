#!/usr/bin/env bash
# run_all.sh — Master launcher for all benchmark experiments
# Usage: bash scripts/run_all.sh [GPU_IDS]
# Example: bash scripts/run_all.sh "0 1 2 3"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${1:-0}"

echo "=========================================="
echo "  Benchmark: Full Dual-Evidence Run"
echo "  GPUs: ${GPU_IDS}"
echo "  $(date)"
echo "=========================================="

# Stage 1: Independent calibration (KGE)
echo ""
echo "--- Stage 1: Independent Calibration (KGE) ---"
bash "${SCRIPT_DIR}/run_calib_kge.sh" "${GPU_IDS}"
echo "Stage 1 done."

# Stage 2: Independent calibration (KGE_LOG)
echo ""
echo "--- Stage 2: Independent Calibration (KGE_LOG) ---"
bash "${SCRIPT_DIR}/run_calib_kge_log.sh" "${GPU_IDS}"
echo "Stage 2 done."

# Stage 3: Parameter learning (KGE)
echo ""
echo "--- Stage 3: Parameter Learning (KGE) ---"
bash "${SCRIPT_DIR}/run_param_learn_kge.sh" "${GPU_IDS}"
echo "Stage 3 done."

# Stage 4: Parameter learning (KGE_LOG)
echo ""
echo "--- Stage 4: Parameter Learning (KGE_LOG) ---"
bash "${SCRIPT_DIR}/run_param_learn_kge_log.sh" "${GPU_IDS}"
echo "Stage 4 done."

# Stage 5: Postprocess (stability + dual-evidence)
echo ""
echo "--- Stage 5: Postprocess (stability + dual-evidence) ---"
cd "${SCRIPT_DIR}/.."
python postprocess.py --output-dir outputs
echo "Stage 5 done."

echo ""
echo "=========================================="
echo "  All stages complete."
echo "  $(date)"
echo "=========================================="
