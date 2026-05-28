#!/usr/bin/env bash
# Independent calibration parallel launcher — KGE_LOG objective
# Usage: bash run_calib_kge_log.sh [gpu_ids...]
# Example: bash run_calib_kge_log.sh 0 1 2 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$BENCH_DIR/logs/calib_kge_log"
mkdir -p "$LOG_DIR"

# GPU list from args or default to single GPU
GPU_IDS=("${@:-0}")
NUM_GPUS=${#GPU_IDS[@]}

MODELS=(
    m01 m02 m03 m04 m05 m06 m07 m08 m09 m10
    m11 m12 m13 m14 m15 m16 m17 m18 m19 m20
    m21 m22 m23 m24 m25 m26 m27 m28 m29 m30
    m31 m32 m33 m34 m35 m36
)

echo "Launching KGE_LOG independent calibration for ${#MODELS[@]} models on GPUs: ${GPU_IDS[*]}"

idx=0
for MODEL in "${MODELS[@]}"; do
    GPU="${GPU_IDS[$((idx % NUM_GPUS))]}"
    LOG_FILE="$LOG_DIR/calib_kge_log_${MODEL}.log"
    echo "  Model=$MODEL GPU=$GPU → $LOG_FILE"
    python "$BENCH_DIR/run_independent_calibration.py" \
        --model-id "$MODEL" \
        --objective KGE_LOG \
        --config "$BENCH_DIR/conf/benchmark_kge_log.yaml" \
        --device "cuda:$GPU" \
        > "$LOG_FILE" 2>&1 &
    idx=$((idx + 1))
done

echo "All KGE_LOG calibration jobs launched. Waiting..."
wait
echo "Done."
