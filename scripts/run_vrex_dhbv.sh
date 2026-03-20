#!/bin/bash
# Run VREx Causal-dPL leave-one-effective-cluster validation.
# Usage:
#   bash scripts/run_vrex_dhbv.sh                # all 7 clusters x 5 seeds + aggregate
#   bash scripts/run_vrex_dhbv.sh A             # one held-out cluster across 5 seeds
#   bash scripts/run_vrex_dhbv.sh A 42          # one held-out cluster, one seed

set -euo pipefail

CONFIG=conf/config_vrex_dhbv.yaml
SCRIPT=train_causal_dpl.py
AGG_SCRIPT=scripts/aggregate_holdout_results.py
CLUSTERS=(A B C D E F G)
SEEDS=(42 123 456 789 1024)
OUTPUT=./outputs/vrex_dhbv_7fold_summary.csv

cd "$(dirname "$0")/.."

run_single() {
    local cluster=$1
    local seed=$2
    echo "========================================"
    echo " Held-out cluster ${cluster} | seed ${seed}"
    echo "========================================"
    uv run python ${SCRIPT} \
        --config ${CONFIG} \
        --holdout ${cluster} \
        --seed ${seed} \
        --mode train_test
}

run_cluster() {
    local cluster=$1
    for seed in "${SEEDS[@]}"; do
        run_single "${cluster}" "${seed}"
    done
}

aggregate_results() {
    uv run python ${AGG_SCRIPT} \
        --input-glob "./outputs/vrex_dhbv_held_out_*_seed*/results_held_out_*_seed*.csv" \
        --output ${OUTPUT} \
        --expect-repeats 5
}

if [ $# -ge 2 ]; then
    run_single "$1" "$2"
elif [ $# -eq 1 ]; then
    run_cluster "$1"
    aggregate_results
else
    for cluster in "${CLUSTERS[@]}"; do
        run_cluster "${cluster}"
    done
    aggregate_results
fi
