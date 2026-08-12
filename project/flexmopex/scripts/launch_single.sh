#!/bin/bash
# Launch a single binary alpha run in background, detached from terminal.
# Usage: bash launch_single.sh <alpha> <seed>

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
    echo "Skipping $RUN_NAME (already complete)"
    exit 0
fi

echo "Launching $RUN_NAME"
setsid python run_model.py \
    --config "$CONFIG" \
    --model-type binary \
    --alpha "$ALPHA" \
    --seed "$SEED" \
    --mode train_test \
    --output-root "$OUTPUT_ROOT" \
    --run-name "$RUN_NAME" \
    > "$LOG_FILE" 2>&1 &

echo "Launched $RUN_NAME (PID: $!)"