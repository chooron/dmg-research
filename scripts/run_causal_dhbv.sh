#!/bin/bash
# Run Causal-dPL group-holdout cross-validation (4 groups, leave-one-out)
# Usage: bash scripts/run_causal_dhbv.sh [holdout_group]
#   holdout_group: 1-4 (default: run all 4)

set -e

CONFIG=conf/config_irm_dhbv.yaml
SCRIPT=train_causal_dpl.py

cd "$(dirname "$0")/.."

run_group() {
    local g=$1
    echo "========================================"
    echo " Holdout Group ${g} / 4"
    echo "========================================"
    python ${SCRIPT} \
        --config ${CONFIG} \
        --holdout ${g} \
        --mode train_test
}

if [ -n "$1" ]; then
    run_group "$1"
else
    for g in 1 2 3 4; do
        run_group ${g}
    done
fi

echo "All runs complete."
