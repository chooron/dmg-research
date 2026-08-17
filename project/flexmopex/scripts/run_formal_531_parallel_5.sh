#!/usr/bin/env bash
# =============================================================================
# Flex-MOPEX Formal 531 100-Epoch 5-Way Parallel Production Launcher
#
# Launches 5 concurrent training runs:
#   1. Base (Fixed w=0)
#   2. Full (Fixed w=1)
#   3. Flex (lambda=0.005, Pure-X35)
#   4. Flex (lambda=0.007, Pure-X35)
#   5. Flex (lambda=0.010, Pure-X35)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXMOPEX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${FLEXMOPEX_DIR}/logs/formal_531_parallel"
PID_FILE="${LOG_DIR}/pids.txt"

mkdir -p "${LOG_DIR}"
rm -f "${PID_FILE}"
touch "${PID_FILE}"

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"

cd "${FLEXMOPEX_DIR}"

launch_job() {
    local name="$1"
    local config="$2"
    local log_file="${LOG_DIR}/${name}.log"

    echo "Launching ${name} on GPU ${GPU_ID} -> log: ${log_file}"
    nohup "${PYTHON}" run_model.py \
        --config "${config}" \
        --output-root "results/formal_531_parallel" \
        --gpu-id "${GPU_ID}" \
        --mode train_test \
        > "${log_file}" 2>&1 &
    local pid=$!
    echo "${name}: PID ${pid}"
    echo "${name}:${pid}:${log_file}" >> "${PID_FILE}"
}

echo "============================================================================="
echo "Starting 5-Way Concurrent Formal 531 Production Runs"
echo "Time: $(date)"
echo "GPU: ${GPU_ID}"
echo "============================================================================="

launch_job "formal_531_base" "conf/config_formal_531_base.yaml"
launch_job "formal_531_full" "conf/config_formal_531_full.yaml"
launch_job "formal_531_flex_lambda0005" "conf/config_formal_531_flex_lambda0005.yaml"
launch_job "formal_531_flex_lambda0007" "conf/config_formal_531_flex_lambda0007.yaml"
launch_job "formal_531_flex_lambda0010" "conf/config_formal_531_flex_lambda0010.yaml"

echo "All 5 jobs launched in parallel!"
echo "PIDs saved to: ${PID_FILE}"
