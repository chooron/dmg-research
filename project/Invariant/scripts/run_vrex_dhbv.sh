#!/usr/bin/env bash
# Run VREx Causal-dPL leave-one-effective-cluster validation.
# Usage:
#   bash scripts/run_vrex_dhbv.sh
#   bash scripts/run_vrex_dhbv.sh A
#   bash scripts/run_vrex_dhbv.sh A 42
#
# Optional environment overrides:
#   CLUSTERS="A B C"
#   SEEDS="42 123 456 789 1024"
#   MODE=train_test
#   EPOCHS=100
#   MC_SAMPLES=100
#   MC_SELECTION_METRIC=mse

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${PROJECT_DIR}/../.." && pwd)

CONFIG=${CONFIG:-${REPO_ROOT}/conf/config_vrex_dhbv.yaml}
SCRIPT=${SCRIPT:-train_causal_dpl.py}
AGG_SCRIPT=${AGG_SCRIPT:-aggregate_holdout_results.py}
MODE=${MODE:-train_test}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs}
OUTPUT_MEAN=${OUTPUT_MEAN:-${OUTPUT_DIR}/vrex_dhbv_7fold_summary_mc_mean.csv}
OUTPUT_AVG=${OUTPUT_AVG:-${OUTPUT_DIR}/vrex_dhbv_7fold_summary_mc_selected.csv}

if [[ -n "${CLUSTERS:-}" ]]; then
    read -r -a CLUSTERS_ARR <<< "${CLUSTERS}"
else
    CLUSTERS_ARR=(A B C D E F G)
fi

if [[ -n "${SEEDS:-}" ]]; then
    read -r -a SEEDS_ARR <<< "${SEEDS}"
else
    SEEDS_ARR=(42 123 456 789 1024)
fi

cd "${PROJECT_DIR}"

build_extra_args() {
    EXTRA_ARGS=()
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
    local cluster=$1
    local seed=$2
    build_extra_args

    echo "========================================"
    echo " VREx | held-out cluster ${cluster} | seed ${seed}"
    echo " Config: ${CONFIG} | mode: ${MODE}"
    echo "========================================"

    uv run python "${SCRIPT}" \
        --config "${CONFIG}" \
        --holdout "${cluster}" \
        --seed "${seed}" \
        --mode "${MODE}" \
        "${EXTRA_ARGS[@]}"
}

run_cluster() {
    local cluster=$1
    for seed in "${SEEDS_ARR[@]}"; do
        run_single "${cluster}" "${seed}"
    done
}

should_aggregate() {
    [[ "${MODE}" == "test" || "${MODE}" == "train_test" ]]
}

aggregate_results() {
    local repeats=${#SEEDS_ARR[@]}

    if ! should_aggregate; then
        echo "Skip aggregation because MODE=${MODE} does not produce evaluation files."
        return
    fi

    uv run python "${AGG_SCRIPT}" \
        --input-glob "./outputs/vrex_dhbv_held_out_*_seed*/results_held_out_*_seed*.csv" \
        --output "${OUTPUT_MEAN}" \
        --expect-repeats "${repeats}"

    uv run python "${AGG_SCRIPT}" \
        --input-glob "./outputs/vrex_dhbv_held_out_*_seed*/results_avg_held_out_*_seed*.csv" \
        --output "${OUTPUT_AVG}" \
        --expect-repeats "${repeats}"
}

if [[ $# -ge 2 ]]; then
    run_single "$1" "$2"
elif [[ $# -eq 1 ]]; then
    run_cluster "$1"
    aggregate_results
else
    for cluster in "${CLUSTERS_ARR[@]}"; do
        run_cluster "${cluster}"
    done
    aggregate_results
fi
