#!/usr/bin/env bash
# Resume only the unfinished CFlex LORO alpha=0.005 runs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
ALPHA="0.005"
OUT_ROOT="results/block3_loro"
LOG_DIR="${LOG_DIR:-logs/block3_loro_resume_alpha0.005}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
TASKS=("4 456" "5 42" "5 123" "5 456" "6 42" "6 123" "6 456")

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-flexmopex}"
mkdir -p "${MPLCONFIGDIR}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local region="$1" seed="$2"
    local run_name="config_dmopex_v1/flex_alpha_0_005_region${region}/seed_${seed}"
    local tag="flex_region${region}_seed${seed}"
    local log_file="${LOG_DIR}/${tag}.log"
    local output_dir="${OUT_ROOT}/${run_name}"

    if [[ -f "${output_dir}/sim/streamflow.npy" ]] && \
       "${PYTHON_BIN}" -c 'import sys,numpy as np; x=np.load(sys.argv[1],mmap_mode="r"); raise SystemExit(0 if np.isfinite(x).all() else 1)' "${output_dir}/sim/streamflow.npy"; then
        log "SKIP ${tag}: finite output exists"
        return 0
    fi

    log "START ${tag}"
    if "${PYTHON_BIN}" run_model.py \
        --model-type flex \
        --alpha "${ALPHA}" \
        --seed "${seed}" \
        --loro-holdout-region "${region}" \
        --output-root "${OUT_ROOT}" \
        --run-name "${run_name}" \
        --gpu-id "${GPU_ID}" \
        --no-verbose >"${log_file}" 2>&1; then
        log "DONE ${tag}"
        return 0
    fi
    log "FAIL ${tag}; see ${log_file}"
    return 1
}

log "Resume CFlex LORO alpha=${ALPHA}; tasks=${#TASKS[@]}; max_parallel=${MAX_PARALLEL}"
fifo="$(mktemp -u)"
mkfifo "${fifo}"
exec 9<>"${fifo}"
rm -f "${fifo}"
for ((i = 0; i < MAX_PARALLEL; i++)); do
    printf 'slot\n' >&9
done

pids=()
failed=0
for task in "${TASKS[@]}"; do
    read -u9
    read -r region seed <<< "${task}"
    (
        status=0
        run_one "${region}" "${seed}" || status=$?
        printf 'slot\n' >&9
        exit "${status}"
    ) &
    pids+=("$!")
done

for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        log "FAILED child_pid=${pids[$i]}"
        failed=1
    fi
done
exec 9>&-

if (( failed )); then
    log "COMPLETE WITH FAILURES"
    exit 1
fi
log "COMPLETE: all missing runs succeeded or were already complete"
