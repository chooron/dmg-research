#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

CONFIG=${CONFIG:-${PROJECT_DIR}/conf/config_dhbv_pub_lstm.yaml}
MODE=${MODE:-train_test}
SEED=${SEED:-42}
LOSS=${LOSS:-}
TEST_EPOCH=${TEST_EPOCH:-50}
TEST_GROUP_ID=${TEST_GROUP_ID:-17}

ARGS=(
    --config "${CONFIG}"
    --mode "${MODE}"
    --seed "${SEED}"
    --test-epoch "${TEST_EPOCH}"
    --test-group-id "${TEST_GROUP_ID}"
)

if [[ -n "${LOSS}" ]]; then
    ARGS+=(--loss "${LOSS}")
fi

echo "bettermodel PUB | model=lstm | mode=${MODE} | seed=${SEED} | test_epoch=${TEST_EPOCH} | test_group_id=${TEST_GROUP_ID}"

cd "${PROJECT_DIR}"
uv run python "${PROJECT_DIR}/run_pub_experiment.py" "${ARGS[@]}"
