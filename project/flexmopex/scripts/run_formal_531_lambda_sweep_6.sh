#!/usr/bin/env bash
# =============================================================================
# Flex-MOPEX Formal 531 100-Epoch 6-Way Lambda Sweep Parallel Launcher
#
# Launches 6 concurrent training runs for the extended lambda spectrum:
#   1. Flex (lambda=0.003, Pure-X35)
#   2. Flex (lambda=0.015, Pure-X35)
#   3. Flex (lambda=0.020, Pure-X35)
#   4. Flex (lambda=0.030, Pure-X35)
#   5. Flex (lambda=0.050, Pure-X35)
#   6. Flex (lambda=0.100, Pure-X35)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXMOPEX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${FLEXMOPEX_DIR}/logs/formal_531_lambda_sweep"
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
echo "Starting 6-Way Concurrent Formal 531 Lambda Sweep Runs"
echo "Time: $(date)"
echo "GPU: ${GPU_ID}"
echo "============================================================================="

launch_job "formal_531_flex_lambda0003" "conf/config_formal_531_flex_lambda0003.yaml"
launch_job "formal_531_flex_lambda0015" "conf/config_formal_531_flex_lambda0015.yaml"
launch_job "formal_531_flex_lambda0020" "conf/config_formal_531_flex_lambda0020.yaml"
launch_job "formal_531_flex_lambda0030" "conf/config_formal_531_flex_lambda0030.yaml"
launch_job "formal_531_flex_lambda0050" "conf/config_formal_531_flex_lambda0050.yaml"
launch_job "formal_531_flex_lambda0100" "conf/config_formal_531_flex_lambda0100.yaml"

echo "All 6 lambda sweep jobs launched in parallel!"
echo "PIDs saved to: ${PID_FILE}"
