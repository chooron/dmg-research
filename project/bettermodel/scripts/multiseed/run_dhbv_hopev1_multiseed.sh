#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

CONFIG=${CONFIG:-${PROJECT_DIR}/conf/config_dhbv_hopev1.yaml}
MODE=${MODE:-train_test}
SEEDS=${SEEDS:-"111 222 333 444 555"}
LOSS=${LOSS:-}
TEST_EPOCH=${TEST_EPOCH:-100}
GPU_ID=${GPU_ID:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${PROJECT_DIR}"

pids=()
labels=()

for SEED in ${SEEDS}; do
    ARGS=(
        --config "${CONFIG}"
        --mode "${MODE}"
        --seed "${SEED}"
        --test-epoch "${TEST_EPOCH}"
        --gpu-id "${GPU_ID}"
    )

    if [[ -n "${LOSS}" ]]; then
        ARGS+=(--loss "${LOSS}")
    fi

    labels+=("seed=${SEED}")
    echo "bettermodel multiseed | model=hopev1 | mode=${MODE} | seed=${SEED} | test_epoch=${TEST_EPOCH} | gpu_id=${GPU_ID}"
    "${PYTHON_BIN}" "${PROJECT_DIR}/run_experiment.py" "${ARGS[@]}" &
    pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "failed: ${labels[$i]}" >&2
        status=1
    fi
done

exit "${status}"
