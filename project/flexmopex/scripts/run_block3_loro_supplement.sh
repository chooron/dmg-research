#!/usr/bin/env bash
# Supplement Block 3 LORO with the missing seed-42 base/full runs and
# alpha=0.007 flex runs.  Outputs are intentionally separated from the
# original alpha=0.01 flex matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

GPU="${GPU:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/block3_loro}"
ALPHA="${LORO_SUPPLEMENT_ALPHA:-0.007}"
read -r -a REGIONS <<< "${LORO_REGIONS:-0 1 2 3 4 5 6}"

export FLEXMOPEX_DATA_DIR="${FLEXMOPEX_DATA_DIR:-/root/autodl-fs}"
export DATA_PATH="${DATA_PATH:-/root/autodl-fs}"
export BASIN_GROUPS_DIR="${BASIN_GROUPS_DIR:-/root/autodl-fs/basin_groups}"
export GAGE_INFO="${GAGE_INFO:-/root/autodl-fs/gage_id.npy}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_inductor_cache}"
export TORCH_COMPILE_DEBUG=0
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${OUT_ROOT}/run_logs"

alpha_tag="${ALPHA//./_}"
RUN_ID="${RUN_ID:-block3_loro_supplement_alpha_${alpha_tag}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${OUT_ROOT}/run_logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

TASKS=()
for region in "${REGIONS[@]}"; do
    TASKS+=("${region} 42 base ${OUT_ROOT}/config_dmopex_v1/base_region${region}/seed_42")
    TASKS+=("${region} 42 full ${OUT_ROOT}/config_dmopex_v1/full_region${region}/seed_42")
    for seed in 42 123 456; do
        TASKS+=("${region} ${seed} flex ${OUT_ROOT}/config_dmopex_v1/flex_alpha_${alpha_tag}_region${region}/seed_${seed}")
    done
done

run_one() {
    local region="$1" seed="$2" model_type="$3" output_dir="$4"
    local tag="${model_type}_region${region}_seed${seed}"
    local log_file="${LOG_DIR}/${tag}.log"

    if [[ -f "${output_dir}/sim/streamflow.npy" ]]; then
        # A failed metrics pass can leave a partially written/NaN prediction
        # file behind.  Only skip outputs that are fully finite.
        if "${PYTHON_BIN}" -c 'import sys, numpy as np; x=np.load(sys.argv[1], mmap_mode="r"); raise SystemExit(0 if np.isfinite(x).all() else 1)' "${output_dir}/sim/streamflow.npy"; then
            echo "[$(date '+%F %T')] SKIP ${tag}: finite output already exists" | tee -a "${log_file}"
            return 0
        fi
        echo "[$(date '+%F %T')] REPAIR ${tag}: existing output contains non-finite values" | tee -a "${log_file}"
    fi

    echo "[$(date '+%F %T')] START ${tag}" | tee "${log_file}"
    if "${PYTHON_BIN}" run_model.py \
        --model-type "${model_type}" \
        --alpha "${ALPHA}" \
        --seed "${seed}" \
        --loro-holdout-region "${region}" \
        --output-root "${OUT_ROOT}" \
        --run-name "${output_dir#${OUT_ROOT}/}" \
        --gpu-id "${GPU}" \
        --no-verbose >> "${log_file}" 2>&1; then
        [[ -f "${output_dir}/sim/streamflow.npy" ]] || {
            echo "[$(date '+%F %T')] FAIL ${tag}: missing streamflow.npy" | tee -a "${log_file}"
            return 1
        }
        echo "[$(date '+%F %T')] OK ${tag}" | tee -a "${log_file}"
    else
        echo "[$(date '+%F %T')] FAIL ${tag}" | tee -a "${log_file}"
        return 1
    fi
}

echo "Block 3 LORO supplement: alpha=${ALPHA}, GPU=${GPU}, max_parallel=${MAX_PARALLEL}"
echo "Logs: ${LOG_DIR}"

fifo="$(mktemp -u)"
mkfifo "${fifo}"
exec 9<>"${fifo}"
rm "${fifo}"
for ((i = 0; i < MAX_PARALLEL; i++)); do printf 'slot\n' >&9; done

pids=()
for task in "${TASKS[@]}"; do
    read -u 9
    (
        read -r region seed model_type output_dir <<< "${task}"
        status=0
        run_one "${region}" "${seed}" "${model_type}" "${output_dir}" || status=$?
        printf 'slot\n' >&9
        exit "${status}"
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
done
exec 9>&-

if [[ "${failed}" -ne 0 ]]; then
    echo "Block 3 LORO supplement completed with failures; inspect ${LOG_DIR}" >&2
    exit 1
fi
echo "Block 3 LORO supplement complete."
