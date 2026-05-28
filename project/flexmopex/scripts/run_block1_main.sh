#!/bin/bash
# Block 1.1: Basic vs Full vs Flex × 5 seeds — parallel execution
# Usage: bash scripts/run_block1_main.sh [GPU_IDS] [MAX_PARALLEL]
#   GPU_IDS:      comma-separated GPU IDs to use, e.g. "0,1,2"  (default: "0")
#   MAX_PARALLEL: max concurrent jobs                             (default: 4)
#
# Examples:
#   bash scripts/run_block1_main.sh 0 4          # 1 GPU, 4 parallel jobs
#   bash scripts/run_block1_main.sh 0,1 8        # 2 GPUs, 8 parallel jobs

set -e

GPU_IDS_STR=${1:-"0"}
MAX_PARALLEL=${2:-4}

IFS=',' read -ra GPUS <<< "$GPU_IDS_STR"
NUM_GPUS=${#GPUS[@]}

SEEDS=(42 123 456 789 1024)
OUT_ROOT="results/block1_main"

mkdir -p "${OUT_ROOT}"

# ── job queue ────────────────────────────────────────────────────────────────
JOB_QUEUE=()
GPU_COUNTER=0

run_exp() {
    local model_type=$1 alpha=$2 seed=$3 gpu=$4
    local tag="${model_type}_alpha${alpha}_seed${seed}"
    local save_path="${OUT_ROOT}/${tag}"
    mkdir -p "${save_path}"
    echo "[$(date +%H:%M:%S)] START  gpu=${gpu}  ${tag}"
    python run_model.py \
        --model-type "${model_type}" \
        --alpha "${alpha}" \
        --seed "${seed}" \
        --gpu-id "${gpu}" \
        > "${save_path}/train.log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE   gpu=${gpu}  ${tag}"
}
export -f run_exp

# ── build task list ──────────────────────────────────────────────────────────
TASKS=()
for seed in "${SEEDS[@]}"; do
    TASKS+=("base          0.0   ${seed}")
    TASKS+=("full          0.0   ${seed}")
    TASKS+=("flex          0.005 ${seed}")
    TASKS+=("flex          0.01  ${seed}")
    TASKS+=("flex          0.03  ${seed}")
done

# ── dispatch with semaphore ──────────────────────────────────────────────────
PIDS=()
JOB_IDX=0

for task in "${TASKS[@]}"; do
    read -r model_type alpha seed <<< "${task}"
    gpu="${GPUS[$((JOB_IDX % NUM_GPUS))]}"

    # throttle: wait until a slot is free
    while [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]]; do
        NEW_PIDS=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                NEW_PIDS+=("${pid}")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]] && sleep 5
    done

    run_exp "${model_type}" "${alpha}" "${seed}" "${gpu}" &
    PIDS+=($!)
    ((JOB_IDX++)) || true
done

# wait for all remaining jobs
for pid in "${PIDS[@]}"; do
    wait "${pid}" || echo "WARNING: job ${pid} exited with error"
done

echo ""
echo "Block 1.1 complete. Results in ${OUT_ROOT}/"
