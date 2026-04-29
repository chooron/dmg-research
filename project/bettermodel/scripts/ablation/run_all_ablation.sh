#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

MODE=${MODE:-train_test}
SEED=${SEED:-42}
LOSS=${LOSS:-}
TEST_EPOCH=${TEST_EPOCH:-100}

SCRIPTS=(
    run_s4d_baseline.sh
    run_s4d_ln.sh
    run_s4d_softsign.sh
    run_s4d_ln_softsign.sh
    run_s5d_conv_only.sh
    run_s5d_conv_bn_softsign.sh
    run_s5d_conv_ln_sigmoid.sh
    run_s5d_full.sh
)

pids=()
labels=()

for script in "${SCRIPTS[@]}"; do
    labels+=("${script}")
    echo "starting ablation ${script} | mode=${MODE} | seed=${SEED} | test_epoch=${TEST_EPOCH}"
    MODE="${MODE}" SEED="${SEED}" LOSS="${LOSS}" TEST_EPOCH="${TEST_EPOCH}" \
        "${SCRIPT_DIR}/${script}" &
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
