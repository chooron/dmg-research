#!/bin/bash
# Master parallel launcher for all experiment blocks
# Usage: bash scripts/run_parallel.sh [MAX_JOBS] [GPU_LIST]
# Examples:
#   bash scripts/run_parallel.sh              # MAX_JOBS=4, GPUs auto-assigned round-robin
#   bash scripts/run_parallel.sh 8            # 8 parallel jobs
#   bash scripts/run_parallel.sh 4 "0,1,2,3" # 4 jobs across 4 GPUs

set -e
MAX_JOBS=${1:-4}
GPU_LIST=${2:-"0"}   # comma-separated list of GPU IDs, e.g. "0,1,2,3"

# Parse GPU list into array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="results/parallel_logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo " Flex-MOPEX Parallel Experiment Launcher"
echo "========================================"
echo "  MAX_JOBS : $MAX_JOBS"
echo "  GPUs     : ${GPUS[*]}"
echo "  Log dir  : $LOG_DIR"
echo "========================================"

# ---- semaphore helpers ----
_JOB_COUNT=0
_JOB_PIDS=()

_wait_for_slot() {
    while [ "$_JOB_COUNT" -ge "$MAX_JOBS" ]; do
        for i in "${!_JOB_PIDS[@]}"; do
            if ! kill -0 "${_JOB_PIDS[$i]}" 2>/dev/null; then
                unset "_JOB_PIDS[$i]"
                _JOB_COUNT=$(( _JOB_COUNT - 1 ))
            fi
        done
        _JOB_PIDS=("${_JOB_PIDS[@]}")   # re-index
        [ "$_JOB_COUNT" -ge "$MAX_JOBS" ] && sleep 2
    done
}

_launch() {
    local label=$1 gpu=$2 script=$3
    shift 3
    _wait_for_slot
    echo "[$(date +%H:%M:%S)] Launching: $label (GPU $gpu)"
    bash "$script" "$gpu" "$@" > "$LOG_DIR/${label}.log" 2>&1 &
    _JOB_PIDS+=($!)
    _JOB_COUNT=$(( _JOB_COUNT + 1 ))
}

_wait_all() {
    echo "[$(date +%H:%M:%S)] Waiting for all jobs to finish..."
    local failed=0
    for pid in "${_JOB_PIDS[@]}"; do
        if wait "$pid"; then
            :
        else
            echo "  [FAILED] PID $pid"
            failed=$(( failed + 1 ))
        fi
    done
    _JOB_PIDS=()
    _JOB_COUNT=0
    return $failed
}

# ---- GPU round-robin helper ----
_GPU_IDX=0
next_gpu() {
    echo "${GPUS[$_GPU_IDX]}"
    _GPU_IDX=$(( (_GPU_IDX + 1) % NUM_GPUS ))
}

# ========================================
# PHASE 1: Block 1 main + alpha path
# ========================================
echo ""
echo ">>> PHASE 1: Block 1 (main comparison + alpha path)"
_launch "block1_main"       "$(next_gpu)" "$SCRIPT_DIR/run_block1_main.sh"
_launch "block1_alpha_path" "$(next_gpu)" "$SCRIPT_DIR/run_block1_alpha_path.sh"
_wait_all
echo ">>> PHASE 1 complete."

# ========================================
# PHASE 2: Block 3 LORO
# ========================================
echo ""
echo ">>> PHASE 2: Block 3 (LORO generalization)"
_launch "block3_loro" "$(next_gpu)" "$SCRIPT_DIR/run_block3_loro.sh"
_wait_all
echo ">>> PHASE 2 complete."

# ========================================
# PHASE 3: Analysis
# ========================================
echo ""
echo ">>> PHASE 3: Post-processing analysis"
bash "$SCRIPT_DIR/run_analysis.sh" all 2>&1 | tee "$LOG_DIR/analysis.log"
echo ">>> PHASE 3 complete."

echo ""
echo "========================================"
echo " All phases complete."
echo " Logs: $LOG_DIR/"
echo "========================================"
