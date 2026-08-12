#!/bin/bash
# Master script: launches all extra alpha runs detached from terminal.
# Run: nohup bash scripts/run_extra_detached.sh &>/dev/null &

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MAX_PARALLEL=2
ALPHAS=(0 0.007 0.05 0.07 0.1)
SEEDS=(42 123 456)
CONFIG="conf/config_binary_v1.yaml"
OUTPUT_ROOT="results/binary_pilot"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_DIR/extra_alpha_master.log"
}

wait_for_slots() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 10
    done
}

log "Starting Binary-Flex extra alpha runs: ${#ALPHAS[@]} alphas x ${#SEEDS[@]} seeds = $((${#ALPHAS[@]} * ${#SEEDS[@]})) runs"
log "Max parallel: $MAX_PARALLEL"

for seed in "${SEEDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        run_name="binary_alpha${alpha}_seed${seed}"
        log_file="$LOG_DIR/${run_name}.log"

        # Skip if already completed (test output dir exists)
        if ls "$OUTPUT_ROOT/$run_name"/test*_Ep50 &>/dev/null 2>&1; then
            log "SKIP $run_name (already complete)"
            continue
        fi

        wait_for_slots
        log "LAUNCH $run_name"
        python run_model.py \
            --config "$CONFIG" \
            --model-type binary \
            --alpha "$alpha" \
            --seed "$seed" \
            --mode train_test \
            --output-root "$OUTPUT_ROOT" \
            --run-name "$run_name" \
            > "$log_file" 2>&1 &
    done
done

wait
log "ALL DONE. Run analysis: python analysis/run_binary_flex_pilot_analysis.py"