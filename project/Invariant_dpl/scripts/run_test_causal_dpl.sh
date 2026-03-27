#!/usr/bin/env bash
# Run causal-dPL checkpoint testing with MC-mean evaluation.
# Usage:
#   bash scripts/run_test_causal_dpl.sh
#   bash scripts/run_test_causal_dpl.sh A
#   bash scripts/run_test_causal_dpl.sh A 20260325
#   bash scripts/run_test_causal_dpl.sh A 20260325 50
#
# Optional environment overrides:
#   CONFIG=./conf/config_vrex_dhbv.yaml
#   CLUSTERS="A B C"
#   SEEDS="20260325 20260326"
#   EPOCH=50
#   MC_SAMPLES=100
#   DATASET=holdout
#   DEVICE=cpu
#   OUTPUT_DIR=/path/to/custom/output_dir
#   SIM_DIR=/path/to/custom/sim_dir

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG=${CONFIG:-${PROJECT_DIR}/conf/config_vrex_dhbv.yaml}
SCRIPT=${SCRIPT:-test_causal_dpl.py}
AGG_SCRIPT=${AGG_SCRIPT:-aggregate_holdout_results.py}
DATASET=${DATASET:-holdout}
EPOCH=${EPOCH:-50}
MC_SAMPLES=${MC_SAMPLES:-10}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-./outputs/causal_dpl_test_${DATASET}_epoch${EPOCH}_mc${MC_SAMPLES}_summary.csv}

if [[ -n "${CLUSTERS:-}" ]]; then
    read -r -a CLUSTERS_ARR <<< "${CLUSTERS}"
else
    CLUSTERS_ARR=(A B C D E F G)
fi

if [[ -n "${SEEDS:-}" ]]; then
    read -r -a SEEDS_ARR <<< "${SEEDS}"
else
    SEEDS_ARR=(20260325)
fi

cd "${PROJECT_DIR}"

build_extra_args() {
    EXTRA_ARGS=(
        --epoch "${EPOCH}"
        --mc-samples "${MC_SAMPLES}"
        --dataset "${DATASET}"
    )
    if [[ -n "${DEVICE:-}" ]]; then
        EXTRA_ARGS+=(--device "${DEVICE}")
    fi
    if [[ -n "${OUTPUT_DIR:-}" ]]; then
        EXTRA_ARGS+=(--output-dir "${OUTPUT_DIR}")
    fi
    if [[ -n "${SIM_DIR:-}" ]]; then
        EXTRA_ARGS+=(--sim-dir "${SIM_DIR}")
    fi
}

run_single() {
    local cluster=$1
    local seed=$2
    local epoch=${3:-${EPOCH}}

    EPOCH=${epoch}
    build_extra_args

    echo "========================================"
    echo " Causal-dPL Test | held-out cluster ${cluster} | seed ${seed} | epoch ${epoch}"
    echo " Config: ${CONFIG} | dataset: ${DATASET} | mc_samples: ${MC_SAMPLES}"
    echo "========================================"

    uv run python "${SCRIPT}" \
        --config "${CONFIG}" \
        --holdout "${cluster}" \
        --seed "${seed}" \
        "${EXTRA_ARGS[@]}"
}

run_cluster() {
    local cluster=$1
    local epoch=${2:-${EPOCH}}
    for seed in "${SEEDS_ARR[@]}"; do
        run_single "${cluster}" "${seed}" "${epoch}"
    done
}

should_aggregate() {
    [[ "${DATASET}" == "holdout" ]] && [[ -z "${OUTPUT_DIR:-}" ]]
}

aggregate_results() {
    local repeats=${#SEEDS_ARR[@]}
    local epoch=${1:-${EPOCH}}
    local summary_output=${SUMMARY_OUTPUT}

    if ! should_aggregate; then
        echo "Skip aggregation because DATASET=${DATASET} or OUTPUT_DIR override disables default glob aggregation."
        return
    fi

    uv run python "${AGG_SCRIPT}" \
        --input-glob "./outputs/*_held_out_*_seed*/test_${DATASET}_epoch${epoch}_mc${MC_SAMPLES}/results_held_out_*_seed*.csv" \
        --output "${summary_output}" \
        --expect-repeats "${repeats}"
}

if [[ $# -ge 3 ]]; then
    run_single "$1" "$2" "$3"
elif [[ $# -eq 2 ]]; then
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
