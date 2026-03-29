#!/usr/bin/env bash
# Run static parameter learning on the fixed 531-basin subset.
# Usage:
#   bash scripts/run_param_learn.sh
#   bash scripts/run_param_learn.sh mc_mlp
#   bash scripts/run_param_learn.sh fast_kan 42
#
# Optional environment overrides:
#   CONFIG=./conf/config_param_learn.yaml
#   MODE=train_test
#   NN_MODEL=mc_mlp
#   SEEDS="42 123 456"
#   EPOCHS=100
#   MC_SAMPLES=100
#   MC_SELECTION_METRIC=mse

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG=${CONFIG:-${PROJECT_DIR}/conf/config_param_learn.yaml}
SCRIPT=${SCRIPT:-train_param_learn.py}
MODE=${MODE:-train_test}
NN_MODEL=${NN_MODEL:-}

if [[ -n "${SEEDS:-}" ]]; then
    read -r -a SEEDS_ARR <<< "${SEEDS}"
else
    SEEDS_ARR=(42)
fi

cd "${PROJECT_DIR}"

build_extra_args() {
    EXTRA_ARGS=(--mode "${MODE}")
    if [[ -n "${EPOCHS:-}" ]]; then
        EXTRA_ARGS+=(--epochs "${EPOCHS}")
    fi
    if [[ -n "${MC_SAMPLES:-}" ]]; then
        EXTRA_ARGS+=(--mc-samples "${MC_SAMPLES}")
    fi
    if [[ -n "${MC_SELECTION_METRIC:-}" ]]; then
        EXTRA_ARGS+=(--mc-selection-metric "${MC_SELECTION_METRIC}")
    fi
}

run_single() {
    local nn_model=$1
    local seed=$2
    build_extra_args

    echo "========================================"
    echo " ParamLearn | nn=${nn_model} | seed=${seed}"
    echo " Config: ${CONFIG} | mode: ${MODE}"
    echo "========================================"

    uv run python "${SCRIPT}" \
        --config "${CONFIG}" \
        --nn-model "${nn_model}" \
        --seed "${seed}" \
        "${EXTRA_ARGS[@]}"
}

run_model() {
    local nn_model=$1
    for seed in "${SEEDS_ARR[@]}"; do
        run_single "${nn_model}" "${seed}"
    done
}

if [[ $# -ge 2 ]]; then
    run_single "$1" "$2"
elif [[ $# -eq 1 ]]; then
    run_model "$1"
elif [[ -n "${NN_MODEL}" ]]; then
    run_model "${NN_MODEL}"
else
    for nn_model in mc_mlp fast_kan; do
        run_model "${nn_model}"
    done
fi
