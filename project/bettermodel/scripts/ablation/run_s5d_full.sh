#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

CONFIG=${CONFIG:-${PROJECT_DIR}/conf/ablation/config_dhbv_ablation_s5d_full.yaml}
MODE=${MODE:-train_test}
SEED=${SEED:-111}
LOSS=${LOSS:-}
TEST_EPOCH=${TEST_EPOCH:-100}

ARGS=(
    --config "${CONFIG}"
    --mode "${MODE}"
    --seed "${SEED}"
    --test-epoch "${TEST_EPOCH}"
)

if [[ -n "${LOSS}" ]]; then
    ARGS+=(--loss "${LOSS}")
fi

echo "bettermodel ablation | model=s5d_full | mode=${MODE} | seed=${SEED} | test_epoch=${TEST_EPOCH}"

cd "${PROJECT_DIR}"
uv run python "${PROJECT_DIR}/run_experiment.py" "${ARGS[@]}"
