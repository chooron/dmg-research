#!/bin/bash
# Launch a single binary run detached from parent process.
# Usage: bash launch_detached.sh <alpha> <seed>

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ALPHA="$1"
SEED="$2"
CONFIG="conf/config_binary_v1.yaml"
OUTPUT_ROOT="results/binary_pilot"
RUN_NAME="binary_alpha${ALPHA}_seed${SEED}"
LOG_FILE="logs/${RUN_NAME}.log"

mkdir -p logs

# Skip if already completed
if ls "$OUTPUT_ROOT/$RUN_NAME"/test*_Ep50 &>/dev/null 2>&1; then
    echo "SKIP: $RUN_NAME (already complete)"
    exit 0
fi

echo "START: $RUN_NAME at $(date '+%H:%M:%S')"

setsid python run_model.py \
    --config "$CONFIG" \
    --model-type binary \
    --alpha "$ALPHA" \
    --seed "$SEED" \
    --mode train_test \
    --output-root "$OUTPUT_ROOT" \
    --run-name "$RUN_NAME" \
    > "$LOG_FILE" 2>&1 </dev/null &

echo "PID: $! for $RUN_NAME"