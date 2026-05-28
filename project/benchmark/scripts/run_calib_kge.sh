#!/usr/bin/env bash
# ============================================================
# Independent Calibration — KGE objective
# Usage: bash run_calib_kge.sh [GPU_IDS...]
# Example: bash run_calib_kge.sh 0 1 2 3
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(dirname "$SCRIPT_DIR")"

# GPU list (default: single GPU 0)
GPU_IDS=("${@:-0}")
N_GPUS=${#GPU_IDS[@]}

LOG_DIR="$BENCHMARK_ROOT/logs/calib_kge"
mkdir -p "$LOG_DIR"

# 36 MARRMoT model IDs
MODELS=(
  m01 m02 m03 m04 m05 m06 m07 m08 m09 m10
  m11 m12 m13 m14 m15 m16 m17 m18 m19 m20
  m21 m22 m23 m24 m25 m26 m27 m28 m29 m30
  m31 m32 m33 m34 m35 m36
)

echo "=== Independent Calibration (KGE) ==="
echo "  GPUs: ${GPU_IDS[*]}"
echo "  Models: ${#MODELS[@]}"
echo "  Log dir: $LOG_DIR"
echo ""

pids=()
idx=0
for MODEL in "${MODELS[@]}"; do
    GPU_ID="${GPU_IDS[$((idx % N_GPUS))]}"
    LOG_FILE="$LOG_DIR/${MODEL}.log"

    echo "  Launching model=$MODEL on cuda:$GPU_ID → $LOG_FILE"
    PYTHONPATH="$BENCHMARK_ROOT/.." python "$BENCHMARK_ROOT/run_independent_calibration.py" \
        --model-id "$MODEL" \
        --objective KGE \
        --config "$BENCHMARK_ROOT/conf/benchmark_kge.yaml" \
        --device "cuda:$GPU_ID" \
        > "$LOG_FILE" 2>&1 &

    pids+=($!)
    idx=$((idx + 1))
done

echo ""
echo "Waiting for ${#pids[@]} jobs..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "  WARNING: job $pid failed"
        failed=$((failed + 1))
    fi
done

if [[ $failed -eq 0 ]]; then
    echo "All calibration jobs (KGE) completed successfully."
else
    echo "$failed job(s) failed. Check logs in $LOG_DIR"
    exit 1
fi
