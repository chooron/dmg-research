#!/bin/bash
# Block 3: Leave-one-region-out (7 hydroclimatic regions, alpha=0.01, 3 seeds)
# Usage: bash scripts/run_block3_loro.sh [GPU_ID] [MAX_PARALLEL]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

GPU=${1:-0}
MAX_PARALLEL=${2:-4}   # max concurrent jobs
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a REGIONS <<< "${LORO_REGIONS:-0 1 2 3 4 5 6}"
read -r -a SEEDS <<< "${LORO_SEEDS:-42 123 456}"
read -r -a MODEL_TYPES <<< "${LORO_MODEL_TYPES:-flex full base}"
ALPHA=0.01
OUT_ROOT="results/block3_loro"

# --- semaphore helpers ---
FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 9<>"$FIFO"
rm "$FIFO"
for ((i=0; i<MAX_PARALLEL; i++)); do printf '%s\n' slot >&9; done

run_exp() {
    local region=$1 seed=$2 model_type=$3
    local tag="${model_type}_region${region}_seed${seed}"
    # Hierarchical: results/block3_loro/{model_type}/region{r}/seed{seed}
    local save_path="${OUT_ROOT}/${model_type}/region${region}/seed${seed}"
    mkdir -p "$save_path"
    echo "[$(date +%H:%M:%S)] Starting: $tag (parallel slot)"
    if ! "${PYTHON_BIN}" run_model.py \
        --model-type "${model_type}" \
        --alpha "${ALPHA}" \
        --seed "${seed}" \
        --loro-holdout-region "${region}" \
        --output-root "${save_path}" \
        --gpu-id "${GPU}" \
        ${RUN_MODEL_EXTRA_ARGS:-} \
        2>&1 | tee "${save_path}/train.log"; then
        echo "[$(date +%H:%M:%S)] Failed: $tag"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] Done: $tag"
}

mkdir -p ${OUT_ROOT}
PIDS=()

for region in "${REGIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for model_type in "${MODEL_TYPES[@]}"; do
            read -u9  # acquire slot
            (
                status=0
                run_exp $region $seed $model_type || status=$?
                printf '%s\n' slot >&9  # release slot
                exit "$status"
            ) &
            PIDS+=($!)
        done
    done
done

# wait for all
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "WARNING: job $pid exited with error"
done

echo "Block 3 (LORO) complete. MAX_PARALLEL=${MAX_PARALLEL}"
