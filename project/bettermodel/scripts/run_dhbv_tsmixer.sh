#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG=${CONFIG:-${PROJECT_DIR}/conf/config_dhbv_tsmixer.yaml}
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

echo "bettermodel | model=tsmixer | mode=${MODE} | seed=${SEED} | test_epoch=${TEST_EPOCH}"

cd "${PROJECT_DIR}"
uv run python "${PROJECT_DIR}/run_experiment.py" "${ARGS[@]}"
