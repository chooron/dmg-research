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
SEED_VALUES=${SEED_VALUES:-${SEEDS:-"111 222 333 444 555"}}
LOSS_VALUES=${LOSS_VALUES:-${LOSSES:-${LOSS:-"HybridNseBatchLoss NseBatchLoss LogNseBatchLoss"}}}
MAX_PARALLEL=${MAX_PARALLEL:-5}

read -r -a SEEDS <<< "${SEED_VALUES}"
read -r -a LOSSES <<< "${LOSS_VALUES}"

cd "${PROJECT_DIR}"

pids=()
active_jobs=0

if (( MAX_PARALLEL < 1 )); then
    echo "MAX_PARALLEL must be >= 1, got ${MAX_PARALLEL}" >&2
    exit 1
fi

wait_for_slot() {
    while (( active_jobs >= MAX_PARALLEL )); do
        if wait -n; then
            active_jobs=$((active_jobs - 1))
        else
            status=$?
            for pid in "${pids[@]}"; do
                kill "${pid}" 2>/dev/null || true
            done
            wait || true
            exit "${status}"
        fi
    done
}

for loss in "${LOSSES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        wait_for_slot

        echo "========================================"
        echo " paper-stack | variant=distributional | loss=${loss} | seed=${seed}"
        echo " config=${CONFIG} | mode=${MODE} | device=${DEVICE} | gpu_id=${GPU_ID} | max_parallel=${MAX_PARALLEL}"
        echo "========================================"

        extra_args=(
            --config "${CONFIG}"
            --variant distributional
            --seed "${seed}"
            --loss "${loss}"
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
        ((active_jobs += 1))
    done
done

while (( active_jobs > 0 )); do
    if wait -n; then
        active_jobs=$((active_jobs - 1))
    else
        status=$?
        for pid in "${pids[@]}"; do
            kill "${pid}" 2>/dev/null || true
        done
        wait || true
        exit "${status}"
    fi
done
