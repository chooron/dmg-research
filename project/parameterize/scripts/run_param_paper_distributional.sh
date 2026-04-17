#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG=${CONFIG:-${PROJECT_DIR}/conf/config_param_paper.yaml}
MODE=${MODE:-train_test}
DEVICE=${DEVICE:-cuda}
GPU_ID=${GPU_ID:-0}
EPOCHS=${EPOCHS:-}
MC_SAMPLES=${MC_SAMPLES:-}
SEEDS=(111 222 333)

cd "${PROJECT_DIR}"

pids=()
for seed in "${SEEDS[@]}"; do
    echo "========================================"
    echo " paper-stack | variant=distributional | seed=${seed}"
    echo " config=${CONFIG} | mode=${MODE} | device=${DEVICE} | gpu_id=${GPU_ID}"
    echo "========================================"

    extra_args=(
        --config "${CONFIG}"
        --variant distributional
        --seed "${seed}"
        --mode "${MODE}"
        --device "${DEVICE}"
        --gpu-id "${GPU_ID}"
    )
    if [[ -n "${EPOCHS:-}" ]]; then
        extra_args+=(--epochs "${EPOCHS}")
    fi
    if [[ -n "${MC_SAMPLES:-}" ]]; then
        extra_args+=(--mc-samples "${MC_SAMPLES}")
    fi

    uv run python "${PROJECT_DIR}/train_param_paper.py" "${extra_args[@]}" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "${pid}"
done
