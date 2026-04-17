#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG=${CONFIG:-${PROJECT_DIR}/conf/config_param_paper.yaml}
MODE=${MODE:-train_test}
EPOCHS=${EPOCHS:-}
MC_SAMPLES=${MC_SAMPLES:-}
SEEDS=(111 222 333)

cd "${PROJECT_DIR}"

pids=()
for seed in "${SEEDS[@]}"; do
    echo "========================================"
    echo " paper-stack | variant=deterministic | seed=${seed}"
    echo " config=${CONFIG} | mode=${MODE}"
    echo "========================================"

    extra_args=(
        --config "${CONFIG}"
        --variant deterministic
        --seed "${seed}"
        --mode "${MODE}"
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
