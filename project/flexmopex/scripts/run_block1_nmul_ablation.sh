#!/bin/bash
# Block 1 ablation: fixed alpha=0.01, seed=42, sweep nmul=(1,8,16,32).
# Runs two jobs at a time in paired order: (1,32), then (8,16).
# Usage:
#   bash scripts/run_block1_nmul_ablation.sh [GPU_IDS]
# Examples:
#   bash scripts/run_block1_nmul_ablation.sh 0
#   bash scripts/run_block1_nmul_ablation.sh 0,1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

GPU_IDS_STR=${1:-"0"}
IFS=',' read -ra GPUS <<< "$GPU_IDS_STR"
NUM_GPUS=${#GPUS[@]}

ALPHA=0.01
SEED=42
CONFIG="conf/config_dmopex_v1.yaml"
OUTPUT_ROOT="results/block1_nmul_ablation"
LOG_DIR="logs/block1_nmul_ablation"
FAILED=0

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-flexmopex}"
mkdir -p "$MPLCONFIGDIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

is_complete() {
    local run_name=$1
    ls "$OUTPUT_ROOT/$run_name"/test*_Ep*/metrics_agg.json >/dev/null 2>&1
}

launch_run() {
    local nmul=$1
    local gpu=$2
    local run_name="flex/alpha${ALPHA}/nmul${nmul}/seed_${SEED}"
    local log_file="$LOG_DIR/flex_alpha${ALPHA}_nmul${nmul}_seed${SEED}.log"

    if is_complete "$run_name"; then
        log "SKIP nmul=${nmul} (already complete)"
        return 0
    fi

    log "START gpu=${gpu} alpha=${ALPHA} nmul=${nmul} seed=${SEED}"
    (
    python run_model.py \
        --config "$CONFIG" \
        --model-type flex \
        --alpha "$ALPHA" \
        --nmul "$nmul" \
        --seed "$SEED" \
        --gpu-id "$gpu" \
        --mode train_test \
        --output-root "$OUTPUT_ROOT" \
        --run-name "$run_name" \
        > "$log_file" 2>&1
    ) &
    PAIR_PIDS+=("$!:$nmul")
}

run_pair() {
    local first=$1
    local second=$2
    local failed=0
    PAIR_PIDS=()
    launch_run "$first" "${GPUS[0]}"
    launch_run "$second" "${GPUS[$((1 % NUM_GPUS))]}"
    for item in "${PAIR_PIDS[@]}"; do
        local pid=${item%%:*}
        local nmul=${item##*:}
        if wait "$pid"; then
            log "DONE nmul=${nmul}"
        else
            log "FAIL nmul=${nmul}; see ${LOG_DIR}/flex_alpha${ALPHA}_nmul${nmul}_seed${SEED}.log"
            failed=1
        fi
    done
    return "$failed"
}

log "Running nmul ablation: alpha=${ALPHA}, seed=${SEED}, output=${OUTPUT_ROOT}"
run_pair 1 32 || FAILED=1
run_pair 8 16 || FAILED=1

log "Collecting nmul ablation summaries"
python scripts/collect_nmul_ablation.py --root "$OUTPUT_ROOT" --alpha "$ALPHA" --seed "$SEED" --nmul 1 8 16 32
if [ "$FAILED" -ne 0 ]; then
    log "DONE with failed runs. Summary: ${OUTPUT_ROOT}/nmul_ablation_summary.md"
    exit 1
fi
log "DONE. Summary: ${OUTPUT_ROOT}/nmul_ablation_summary.md"
