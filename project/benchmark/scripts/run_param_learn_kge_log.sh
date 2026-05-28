#!/usr/bin/env bash
# Parallel parameter learning runner — KGE_LOG objective
# Usage: bash run_param_learn_kge_log.sh [GPU_IDS...]
# Example: bash run_param_learn_kge_log.sh 0 1 2 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(dirname "$SCRIPT_DIR")"

GPU_IDS=("${@:-0}")
NUM_GPUS=${#GPU_IDS[@]}

MODELS=(
    m01 m02 m03 m04 m05 m06 m07 m08 m09 m10
    m11 m12 m13 m14 m15 m16 m17 m18 m19 m20
    m21 m22 m23 m24 m25 m26 m27 m28 m29 m30
    m31 m32 m33 m34 m35 m36
)

LOG_DIR="${BENCHMARK_ROOT}/logs/param_learn_kge_log"
mkdir -p "$LOG_DIR"

echo "[$(date)] Starting parameter learning KGE_LOG for ${#MODELS[@]} models on GPUs: ${GPU_IDS[*]}"

idx=0
pids=()

for MODEL in "${MODELS[@]}"; do
    GPU=${GPU_IDS[$((idx % NUM_GPUS))]}
    LOG="${LOG_DIR}/${MODEL}.log"

    echo "[$(date)] Launching ${MODEL} on GPU ${GPU} → ${LOG}"
    CUDA_VISIBLE_DEVICES="${GPU}" python "${BENCHMARK_ROOT}/run_parameter_learning.py" \
        --model-id "${MODEL}" \
        --objective KGE_LOG \
        --config "${BENCHMARK_ROOT}/conf/param_learning_kge_log.yaml" \
        --gpu-id "${GPU}" \
        > "${LOG}" 2>&1 &

    pids+=($!)
    idx=$((idx + 1))

    # Throttle: wait for a slot every NUM_GPUS launches
    if (( idx % NUM_GPUS == 0 )); then
        wait "${pids[@]}"
        pids=()
        echo "[$(date)] Batch of ${NUM_GPUS} jobs done."
    fi
done

# Wait for any remaining jobs
if [ ${#pids[@]} -gt 0 ]; then
    wait "${pids[@]}"
fi

echo "[$(date)] All parameter learning KGE_LOG jobs completed."
