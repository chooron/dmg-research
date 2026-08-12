#!/usr/bin/env bash
set -uo pipefail

cd /root/dmg-research/project/flexmopex

export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/dmg-research:${PYTHONPATH:-}
export PYTHON_BIN=/root/miniconda3/bin/python
export FLEXMOPEX_DATA_DIR=/root/autodl-fs
export DATA_PATH=/root/autodl-fs
export BASIN_GROUPS_DIR=/root/autodl-fs/basin_groups
export GAGE_INFO=/root/autodl-fs/gage_id.npy
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_inductor_cache}
export TORCH_COMPILE_DEBUG=0
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

GPU=${GPU:-0}
ALPHA=${LORO_ALPHA:-0.01}
OUT_ROOT=${OUT_ROOT:-results/block3_loro}
MAX_PARALLEL=${MAX_PARALLEL:-4}
RETRY_MAX_PARALLEL=${RETRY_MAX_PARALLEL:-1}
SEEDS=(123 456)
MODELS=(base full flex)
REGIONS=(0 1 2 3 4 5 6)
RUN_ID=${RUN_ID:-block3_loro_repeat2_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${OUT_ROOT}/run_logs/${RUN_ID}
mkdir -p "${LOG_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

TASKS=()
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for model in "${MODELS[@]}"; do
      TASKS+=("${region} ${seed} ${model}")
    done
  done
done

run_one_task() {
  local region="$1" seed="$2" model="$3"
  local tag="${model}_region${region}_seed${seed}"
  local out_npy="${OUT_ROOT}/config_dmopex_v1/${model}_region${region}/seed_${seed}/sim/streamflow.npy"
  local log_file="${LOG_DIR}/${tag}.log"

  if [[ -f "${out_npy}" ]]; then
    echo "[$(date '+%F %T')] SKIP existing ${tag}" | tee -a "${log_file}"
    return 0
  fi

  echo "[$(date '+%F %T')] START ${tag}" | tee "${log_file}"
  "${PYTHON_BIN}" run_batch_loro.py \
    --region "${region}" \
    --seeds "${seed}" \
    --model-types "${model}" \
    --gpu-id "${GPU}" \
    --alpha "${ALPHA}" \
    --output-root "${OUT_ROOT}" \
    >> "${log_file}" 2>&1
  local status=$?

  if [[ "${status}" -eq 0 && -f "${out_npy}" ]]; then
    echo "[$(date '+%F %T')] OK ${tag}" | tee -a "${log_file}"
    return 0
  fi

  echo "[$(date '+%F %T')] FAIL ${tag} status=${status}" | tee -a "${log_file}"
  return 1
}

run_task_set() {
  local max_parallel="$1" failed_file="$2"
  shift 2
  local task_list=("$@")
  : > "${failed_file}"

  local fifo
  fifo=$(mktemp -u)
  mkfifo "${fifo}"
  exec 9<>"${fifo}"
  rm "${fifo}"
  for ((i = 0; i < max_parallel; i++)); do
    printf '%s\n' slot >&9
  done

  local pids=()
  for task in "${task_list[@]}"; do
    read -u9
    (
      read -r region seed model <<< "${task}"
      if ! run_one_task "${region}" "${seed}" "${model}"; then
        printf '%s\n' "${task}" >> "${failed_file}"
      fi
      printf '%s\n' slot >&9
    ) &
    pids+=($!)
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done
  exec 9>&-
}

echo "========================================================"
echo "Block3 LORO matrix run: ${RUN_ID}"
echo "GPU=${GPU} MAX_PARALLEL=${MAX_PARALLEL} RETRY_MAX_PARALLEL=${RETRY_MAX_PARALLEL}"
echo "Seeds: ${SEEDS[*]}"
echo "Models: ${MODELS[*]}"
echo "Regions: ${REGIONS[*]}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "========================================================"

FIRST_FAILED="${LOG_DIR}/failed_first.txt"
FINAL_FAILED="${LOG_DIR}/failed_final.txt"
run_task_set "${MAX_PARALLEL}" "${FIRST_FAILED}" "${TASKS[@]}"

if [[ -s "${FIRST_FAILED}" ]]; then
  echo "[$(date '+%F %T')] First pass failures; retrying serially:"
  cat "${FIRST_FAILED}"
  mapfile -t RETRY_TASKS < "${FIRST_FAILED}"
  run_task_set "${RETRY_MAX_PARALLEL}" "${FINAL_FAILED}" "${RETRY_TASKS[@]}"
else
  : > "${FINAL_FAILED}"
fi

if [[ -s "${FINAL_FAILED}" ]]; then
  echo "[$(date '+%F %T')] FINAL FAILURES:"
  cat "${FINAL_FAILED}"
  exit 1
fi

echo "[$(date '+%F %T')] Block3 LORO matrix complete: all tasks succeeded or already existed."
