#!/usr/bin/env bash
# =============================================================================
# Flex-MOPEX 30-Epoch 7-Way BCE / Counterfactual Ablation Parallel Launcher
#
# Runs:
#   A: Shared backbone, BCE OFF, lambda=0.007
#   B: Independent pure-X35, BCE OFF, lambda=0.007
#   C: Shared backbone, BCE ON, lambda=0.007
#   D: Independent pure-X35, BCE ON, lambda=0.007
#   E: Independent pure-X35, BCE OFF, lambda=0.003
#   F: Independent pure-X35, BCE ON, lambda=0.003
#   G: Independent pure-X35, BCE OFF, lambda=0.000
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXMOPEX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${FLEXMOPEX_DIR}/logs/bce_ablation_30ep"
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
        --output-root "results/bce_ablation_30ep" \
        --gpu-id "${GPU_ID}" \
        --mode train_test \
        > "${log_file}" 2>&1 &
    local pid=$!
    echo "${name}: PID ${pid}"
    echo "${name}:${pid}:${log_file}" >> "${PID_FILE}"
}

echo "============================================================================="
echo "Starting 7-Way Concurrent BCE / Counterfactual Ablation Runs (30 Epochs, Seed 42)"
echo "Time: $(date)"
echo "GPU: ${GPU_ID}"
echo "============================================================================="

launch_job "exp_A_shared_nobce_lambda0007" "conf/config_ablation_A_shared_nobce_lambda0007.yaml"
launch_job "exp_B_indep_nobce_lambda0007" "conf/config_ablation_B_indep_nobce_lambda0007.yaml"
launch_job "exp_C_shared_bce_lambda0007" "conf/config_ablation_C_shared_bce_lambda0007.yaml"
launch_job "exp_D_indep_bce_lambda0007" "conf/config_ablation_D_indep_bce_lambda0007.yaml"
launch_job "exp_E_indep_nobce_lambda0003" "conf/config_ablation_E_indep_nobce_lambda0003.yaml"
launch_job "exp_F_indep_bce_lambda0003" "conf/config_ablation_F_indep_bce_lambda0003.yaml"
launch_job "exp_G_indep_nobce_lambda0000" "conf/config_ablation_G_indep_nobce_lambda0000.yaml"

echo "All 7 ablation jobs launched in parallel!"
echo "PIDs saved to: ${PID_FILE}"
