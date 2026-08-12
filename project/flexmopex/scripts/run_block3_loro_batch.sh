#!/bin/bash
# Block 3 LORO – batch mode: one Python process per region (all seeds+model_types inside).
#
# Optimization vs run_block3_loro.sh:
#   1. Each region runs run_batch_loro.py (one process) instead of N×M separate
#      `python run_model.py` calls → torch.compile fires once and its kernels are
#      reused across seeds/model_types within the process.
#   2. TORCHINDUCTOR_CACHE_DIR is set so even across regions the compiled kernels
#      are read from disk (no recompilation on subsequent processes).
#   3. Regions are serialised by default (MAX_PARALLEL=1) to avoid GPU memory
#      contention. Set MAX_PARALLEL=2+ only if the GPU has enough VRAM.
#
# Usage:
#   bash scripts/run_block3_loro_batch.sh [GPU_ID] [MAX_PARALLEL]
#
# Environment variables:
#   LORO_REGIONS      – space-separated region indices (default: 0 1 2 3 4 5 6)
#   LORO_SEEDS        – space-separated seeds       (default: 42)
#   LORO_MODEL_TYPES  – space-separated model types (default: flex full base)
#   PYTHON_BIN        – Python interpreter          (default: python)
#   TORCHINDUCTOR_CACHE_DIR – compile cache dir     (default: /tmp/torch_inductor_cache)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

GPU=${1:-0}
MAX_PARALLEL=${2:-1}   # keep =1 to avoid GPU memory contention
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a REGIONS     <<< "${LORO_REGIONS:-0 1 2 3 4 5 6}"
read -r -a SEEDS       <<< "${LORO_SEEDS:-42}"
read -r -a MODEL_TYPES <<< "${LORO_MODEL_TYPES:-flex full base}"
ALPHA="${LORO_ALPHA:-0.01}"
OUT_ROOT="${OUT_ROOT:-results/block3_loro}"

# Persistent compile cache – shared across all processes on this machine
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_inductor_cache}"
export TORCH_COMPILE_DEBUG=0
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

echo "========================================================"
echo " Block 3 LORO (batch mode)"
echo "  GPU            : ${GPU}"
echo "  MAX_PARALLEL   : ${MAX_PARALLEL}"
echo "  Regions        : ${REGIONS[*]}"
echo "  Seeds          : ${SEEDS[*]}"
echo "  Model types    : ${MODEL_TYPES[*]}"
echo "  Alpha          : ${ALPHA}"
echo "  Output root    : ${OUT_ROOT}"
echo "  Compile cache  : ${TORCHINDUCTOR_CACHE_DIR}"
echo "========================================================"

mkdir -p "${OUT_ROOT}"

# --- semaphore helpers (token bucket for MAX_PARALLEL) ---
FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 9<>"$FIFO"
rm "$FIFO"
for ((i=0; i<MAX_PARALLEL; i++)); do printf '%s\n' slot >&9; done

PIDS=()
FAILED=()

run_region() {
    local region=$1
    local log_file="${OUT_ROOT}/batch_region${region}.log"

    # Skip if all model_type/seed combos already have streamflow.npy
    local all_done=true
    for mt in "${MODEL_TYPES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            local out_npy="${OUT_ROOT}/config_dmopex_v1/${mt}_region${region}/seed_${seed}/sim/streamflow.npy"
            if [[ ! -f "${out_npy}" ]]; then
                all_done=false
                break 2
            fi
        done
    done
    if [[ "${all_done}" == "true" ]]; then
        echo "[$(date +%H:%M:%S)] Region ${region} already complete, skipping."
        return 0
    fi

    echo "[$(date +%H:%M:%S)] Starting region ${region} (process $$)"
    if "${PYTHON_BIN}" run_batch_loro.py \
        --region "${region}" \
        --seeds  "${SEEDS[@]}" \
        --model-types "${MODEL_TYPES[@]}" \
        --gpu-id "${GPU}" \
        --alpha  "${ALPHA}" \
        --output-root "${OUT_ROOT}" \
        2>&1 | tee "${log_file}"; then
        echo "[$(date +%H:%M:%S)] Region ${region} done"
    else
        echo "[$(date +%H:%M:%S)] Region ${region} FAILED – see ${log_file}" >&2
        return 1
    fi
}

for region in "${REGIONS[@]}"; do
    read -u9   # acquire slot
    (
        status=0
        run_region "${region}" || status=$?
        printf '%s\n' slot >&9   # release slot
        exit "$status"
    ) &
    PIDS+=($!)
done

# Wait for all sub-processes
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED+=("$pid")
    fi
done

echo ""
echo "========================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo " Block 3 LORO (batch) complete. All regions succeeded."
else
    echo " Block 3 LORO (batch) finished with ${#FAILED[@]} failed region(s)."
    echo " Failed PIDs: ${FAILED[*]}"
    exit 1
fi
echo "========================================================"
