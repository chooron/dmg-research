#!/bin/bash
# Binary-Flex-MOPEX full pilot: 5 alphas x 3 seeds, max 2 parallel jobs.
# Run from project/flexmopex/:
#   bash scripts/run_binary_full_pilot.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MAX_PARALLEL=2
ALPHAS=(0.001 0.003 0.005 0.01 0.03)
SEEDS=(42 123 456)
CONFIG="conf/config_binary_v1.yaml"
OUTPUT_ROOT="outputs/binary_pilot"

mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_slots() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 10
    done
}

log "Starting Binary-Flex full pilot: ${#ALPHAS[@]} alphas x ${#SEEDS[@]} seeds = $((${#ALPHAS[@]} * ${#SEEDS[@]})) runs"
log "Max parallel: $MAX_PARALLEL"

for seed in "${SEEDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        run_name="binary_alpha${alpha}_seed${seed}"
        log_file="logs/${run_name}.log"

        # Skip if already completed (test output dir exists)
        if ls "$OUTPUT_ROOT/$run_name"/test*_Ep50 &>/dev/null 2>&1; then
            log "Skipping $run_name (already complete)"
            continue
        fi

        wait_for_slots
        log "Launching $run_name"
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
log "All runs complete."
log "Run analysis: python analysis/run_binary_flex_pilot_analysis.py"
