#!/bin/bash
# Block 1.2: Alpha regularization path (flex only)
# Usage: bash scripts/run_block1_alpha_path.sh [MAX_JOBS] [GPU_LIST]
# Examples:
#   bash scripts/run_block1_alpha_path.sh 4
#   bash scripts/run_block1_alpha_path.sh 4 "0,1,2,3"
set -e

MAX_JOBS=${1:-4}
GPU_LIST=${2:-"0"}
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

KEY_ALPHAS=(0.005 0.01 0.03)
PATH_ALPHAS=(0.0 0.001 0.003 0.007 0.05 0.07 0.1)
KEY_SEEDS=(42 123 456 789 1024)
PATH_SEEDS=(42 123 456)
OUT_ROOT="results/block1_alpha_path"

mkdir -p "${OUT_ROOT}"

# Semaphore: limit concurrent jobs
JOB_COUNT=0
JOB_IDX=0

run_exp() {
    local alpha=$1 seed=$2
    local gpu_idx=$(( JOB_IDX % NUM_GPUS ))
    local gpu=${GPUS[$gpu_idx]}
    local tag="flex_alpha${alpha}_seed${seed}"
    local save_path="${OUT_ROOT}/${tag}"
    mkdir -p "${save_path}"
    echo "[$(date +%H:%M:%S)] Starting: $tag (GPU ${gpu})"
    python run_model.py \
        --config-name config_experiment \
        model_type=flex \
        loss_function.aic_alpha=${alpha} \
        random_seed=${seed} \
        save_path=${save_path}/ \
        trained_model=${save_path}/model/ \
        gpu_id=${gpu} \
        2>&1 | tee "${save_path}/train.log" &
}

wait_for_slot() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_JOBS ]]; do
        sleep 5
    done
}

# Key alphas: 5 seeds
for alpha in "${KEY_ALPHAS[@]}"; do
    for seed in "${KEY_SEEDS[@]}"; do
        wait_for_slot
        run_exp $alpha $seed
        JOB_IDX=$(( JOB_IDX + 1 ))
    done
done

# Full path alphas: 3 seeds
for alpha in "${PATH_ALPHAS[@]}"; do
    for seed in "${PATH_SEEDS[@]}"; do
        wait_for_slot
        run_exp $alpha $seed
        JOB_IDX=$(( JOB_IDX + 1 ))
    done
done

wait
echo "[$(date +%H:%M:%S)] Block 1.2 (alpha path) complete."
