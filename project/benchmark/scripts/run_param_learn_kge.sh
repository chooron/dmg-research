#!/usr/bin/env bash
# Parameter learning – KGE objective, parallel over 36 models
# Usage: bash run_param_learn_kge.sh [gpu_ids...]
# Example: bash run_param_learn_kge.sh 0 1 2 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/param_learn_kge"
mkdir -p "$LOG_DIR"

GPU_IDS=("${@:-0}")
N_GPUS="${#GPU_IDS[@]}"

MODELS=(
  m01 m02 m03 m04 m05 m06 m07 m08 m09 m10
  m11 m12 m13 m14 m15 m16 m17 m18 m19 m20
  m21 m22 m23 m24 m25 m26 m27 m28 m29 m30
  m31 m32 m33 m34 m35 m36
)

echo "=== Parameter Learning KGE: ${#MODELS[@]} models on ${N_GPUS} GPU(s) ==="

idx=0
for MODEL in "${MODELS[@]}"; do
    GPU="${GPU_IDS[$((idx % N_GPUS))]}"
    LOG="$LOG_DIR/${MODEL}.log"
    echo "  Launching $MODEL on GPU $GPU → $LOG"
    CUDA_VISIBLE_DEVICES="$GPU" python "$PROJECT_DIR/run_parameter_learning.py" \
        --model-id "$MODEL" \
        --objective KGE \
        --config "$PROJECT_DIR/conf/param_learning_kge.yaml" \
        --device "cuda:0" \
        > "$LOG" 2>&1 &
    idx=$((idx + 1))
    # Limit concurrency to number of GPUs
    if (( idx % N_GPUS == 0 )); then
        wait
    fi
done
wait
echo "=== All param-learn KGE jobs done ==="
