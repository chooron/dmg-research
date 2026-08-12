#!/usr/bin/env bash
# CFlex nmul ablation at lambda=0.007, seeds 42/123/456.
#
# Scheduling: for each seed, keep four processes active together in the
# balanced order (nmul 1,32) and (nmul 8,16).  This keeps GPU utilization
# high while pairing low- and high-memory variants.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_STR="${GPU_IDS:-0}"
ALPHA="0.007"
CONFIG="conf/config_dmopex_v1.yaml"
OUTPUT_ROOT="results/block1_nmul_ablation"
LOG_DIR="${LOG_DIR:-logs/block1_nmul_ablation_lambda0.007}"
SEEDS=(42 123 456)
NMULS=(1 8 16 32)

IFS=',' read -r -a GPUS <<< "${GPU_IDS_STR}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-flexmopex}"
mkdir -p "${MPLCONFIGDIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

is_complete() {
    local run_name="$1"
    find "${OUTPUT_ROOT}/${run_name}" -path '*/test*_Ep*/metrics_agg.json' -type f -print -quit 2>/dev/null | grep -q .
}

gpu_for_index() {
    local index="$1"
    echo "${GPUS[$((index % ${#GPUS[@]}))]}"
}

run_one() {
    local nmul="$1" seed="$2" gpu="$3"
    local run_name="flex/lambda0.007/nmul${nmul}/seed${seed}"
    local tag="nmul${nmul}_seed${seed}"
    local log_file="${LOG_DIR}/${tag}.log"

    if is_complete "${run_name}"; then
        log "SKIP ${tag}: complete"
        return 0
    fi

    log "START ${tag} gpu=${gpu}"
    "${PYTHON_BIN}" run_model.py \
        --config "${CONFIG}" \
        --model-type flex \
        --alpha "${ALPHA}" \
        --nmul "${nmul}" \
        --seed "${seed}" \
        --gpu-id "${gpu}" \
        --mode train_test \
        --output-root "${OUTPUT_ROOT}" \
        --run-name "${run_name}" \
        >"${log_file}" 2>&1
    log "DONE ${tag}"
}

run_seed_wave() {
    local seed="$1"
    local pids=() tags=() index=0 failed=0
    local nmul gpu
    log "SEED WAVE seed=${seed}: nmul 1+32 and 8+16 (4 processes)"
    for nmul in 1 32 8 16; do
        gpu="$(gpu_for_index "${index}")"
        run_one "${nmul}" "${seed}" "${gpu}" &
        pids+=("$!")
        tags+=("nmul${nmul}_seed${seed}")
        index=$((index + 1))
    done
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            log "FAIL ${tags[$i]}"
            failed=1
        fi
    done
    return "${failed}"
}

log "CFlex nmul ablation: lambda=${ALPHA}, seeds=${SEEDS[*]}, output=${OUTPUT_ROOT}"
FAILED=0
for seed in "${SEEDS[@]}"; do
    run_seed_wave "${seed}" || FAILED=1
done

if (( FAILED )); then
    log "COMPLETE WITH FAILURES; inspect ${LOG_DIR}"
    exit 1
fi
log "COMPLETE: all requested runs succeeded or were already complete"
