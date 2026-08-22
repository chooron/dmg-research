#!/usr/bin/env bash
# =============================================================================
# Flex-MOPEX Formal 531 Multi-Seed Stability Launcher (Seeds 43 & 44)
#
# Runs:
#   1. Flex (lambda=0.003, Seed 43)
#   2. Flex (lambda=0.003, Seed 44)
#   3. Flex (lambda=0.007, Seed 43)
#   4. Flex (lambda=0.007, Seed 44)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXMOPEX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${FLEXMOPEX_DIR}/logs/formal_531_multiseed"
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
        --output-root "results/formal_531_multiseed" \
        --gpu-id "${GPU_ID}" \
        --mode train_test \
        > "${log_file}" 2>&1 &
    local pid=$!
    echo "${name}: PID ${pid}"
    echo "${name}:${pid}:${log_file}" >> "${PID_FILE}"
}

echo "============================================================================="
echo "Starting 4-Way Multi-Seed Stability Runs (λ=0.003 & λ=0.007, Seeds 43 & 44)"
echo "Time: $(date)"
echo "GPU: ${GPU_ID}"
echo "============================================================================="

launch_job "formal_531_flex_lambda0003_seed43" "conf/config_formal_531_flex_lambda0003_seed43.yaml"
launch_job "formal_531_flex_lambda0003_seed44" "conf/config_formal_531_flex_lambda0003_seed44.yaml"
launch_job "formal_531_flex_lambda0007_seed43" "conf/config_formal_531_flex_lambda0007_seed43.yaml"
launch_job "formal_531_flex_lambda0007_seed44" "conf/config_formal_531_flex_lambda0007_seed44.yaml"

echo "All 4 multi-seed jobs launched in parallel!"
echo "PIDs saved to: ${PID_FILE}"
