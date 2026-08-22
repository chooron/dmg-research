#!/usr/bin/env bash
# Staged CAMELS-671 formal runner. No result directory is ever overwritten.
# Usage: GPU_IDS=0,1 bash scripts/run_formal_671_staged.sh unified|reference|lopo|nmul|loro|dflex
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_STR="${GPU_IDS:-0}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS_STR}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPUS[@]}}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-1}"
SCHEDULER_MODE="${SCHEDULER_MODE:-round_robin}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
MIN_EPOCHS="${MIN_EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-0.0001}"
if [[ "${PATIENCE}" -ne 10 ]]; then
  echo "PATIENCE is fixed at 10 for the formal 671 protocol" >&2
  exit 2
fi
if ! [[ "${PROCESSES_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROCESSES_PER_GPU must be a positive integer" >&2
  exit 2
fi
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"
NMUL_SMALL_PROCESSES_PER_GPU="${NMUL_SMALL_PROCESSES_PER_GPU:-5}"
NMUL16_PROCESSES_PER_GPU="${NMUL16_PROCESSES_PER_GPU:-3}"
NMUL32_PROCESSES_PER_GPU="${NMUL32_PROCESSES_PER_GPU:-2}"
for nmul_slot_value in "${NMUL_SMALL_PROCESSES_PER_GPU}" "${NMUL16_PROCESSES_PER_GPU}" "${NMUL32_PROCESSES_PER_GPU}"; do
  if ! [[ "${nmul_slot_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "nmul phase process counts must be positive integers" >&2
    exit 2
  fi
done
STAGE="${1:-}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
DRY_RUN="${DRY_RUN:-0}"
if [[ -z "${STAGE}" ]]; then
  echo "Usage: GPU_IDS=0,1 $0 unified|reference|lopo|dflex|nmul|loro" >&2
  exit 2
fi

if [[ "${STAGE}" == "dflex" ]]; then
  echo "DFlex is blocked: no tracked formal DFlex config/provenance, and the legacy L0 objective conflicts with the universal CF/BCE contract." >&2
  exit 3
fi

run_one() {
  local gpu="$1" output_root="$2" run_name="$3" config="$4"
  shift 4
  if [[ "${RESULTS_ROOT}" != "results" && "${output_root}" == results/* ]]; then
    output_root="${RESULTS_ROOT}/${output_root#results/}"
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN stage=${STAGE} gpu=${gpu} run=${run_name} config=${config} args=$*"
    return 0
  fi
  local run_dir="${output_root}/${run_name}"
  local resume_existing=0
  local retry_existing=0
  if [[ "${FORCE_OVERWRITE}" != "1" ]] && [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -print -quit)" ]]; then
    if [[ -f "${run_dir}/early_stopping.json" && -f "${run_dir}/model/last_checkpoint.pt" ]] && { [[ -f "${run_dir}/sim/metrics.json" ]] || compgen -G "${run_dir}/test*/metrics.json" >/dev/null; }; then
      echo "SKIP complete ${run_name}"
      return 0
    fi
    if [[ "${RESUME_INCOMPLETE}" == "1" ]]; then
      if [[ -f "${run_dir}/model/last_checkpoint.pt" ]]; then
        resume_existing=1
      else
        retry_existing=1
      fi
    else
      echo "REFUSE non-empty incomplete output: ${run_dir}" >&2
      return 1
    fi
  fi
  mkdir -p "${run_dir}"
  if [[ "${resume_existing}" -eq 1 ]]; then
    echo "RESUME stage=${STAGE} gpu=${gpu} run=${run_name}"
    exec 9>>"${run_dir}/train.log"
  elif [[ "${retry_existing}" -eq 1 ]]; then
    echo "RETRY stage=${STAGE} gpu=${gpu} run=${run_name}"
    exec 9>>"${run_dir}/train.log"
  else
    echo "START stage=${STAGE} gpu=${gpu} run=${run_name}"
    exec 9>"${run_dir}/train.log"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" run_model.py \
    --config "${config}" \
    --mode train_test \
    --gpu-id 0 \
    --output-root "${output_root}" \
    --run-name "${run_name}" \
    --epochs "${MAX_EPOCHS}" \
    --min-epochs "${MIN_EPOCHS}" \
    --early-stop-patience "${PATIENCE}" \
    --early-stop-min-delta "${MIN_DELTA}" \
    "$@" >&9 2>&9
  exec 9>&-
  echo "DONE stage=${STAGE} run=${run_name}"
}

run_wave_round_robin() {
  local -n jobs_ref="$1"
  local -a pids=() labels=()
  local idx=0
  local failures=0
  for job in "${jobs_ref[@]}"; do
    while [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]]; do
      local -a next_pids=() next_labels=()
      for i in "${!pids[@]}"; do
        if kill -0 "${pids[$i]}" 2>/dev/null; then
          next_pids+=("${pids[$i]}")
          next_labels+=("${labels[$i]}")
        else
          if ! wait "${pids[$i]}"; then
            failures=1
          fi
        fi
      done
      pids=("${next_pids[@]}")
      labels=("${next_labels[@]}")
      [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]] && sleep 2
    done
    IFS=$'\t' read -r output_root run_name config extra <<< "${job}"
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    # shellcheck disable=SC2086
    run_one "${gpu}" "${output_root}" "${run_name}" "${config}" ${extra} &
    pids+=("$!")
    labels+=("${run_name}")
    idx=$((idx + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=1
    fi
  done
  return "${failures}"
}

run_wave_per_gpu() {
  local -n jobs_ref="$1"
  local -a pids=() labels=() pid_gpus=()
  local next_job=0
  local failures=0

  launch_slot() {
    local gpu="$1"
    local job output_root run_name config extra
    IFS=$'\t' read -r output_root run_name config extra <<< "${jobs_ref[$next_job]}"
    # shellcheck disable=SC2086
    run_one "${gpu}" "${output_root}" "${run_name}" "${config}" ${extra} &
    pids+=("$!")
    labels+=("${run_name}")
    pid_gpus+=("${gpu}")
    next_job=$((next_job + 1))
  }

  for gpu in "${GPUS[@]}"; do
    for ((slot = 0; slot < PROCESSES_PER_GPU && next_job < ${#jobs_ref[@]}; slot++)); do
      launch_slot "${gpu}"
    done
  done

  while [[ "${#pids[@]}" -gt 0 ]]; do
    local progress=0
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        if ! wait "${pids[$i]}"; then
          failures=1
        fi
        local gpu="${pid_gpus[$i]}"
        unset 'pids[i]' 'labels[i]' 'pid_gpus[i]'
        pids=("${pids[@]}")
        labels=("${labels[@]}")
        pid_gpus=("${pid_gpus[@]}")
        if [[ "${next_job}" -lt "${#jobs_ref[@]}" ]]; then
          launch_slot "${gpu}"
        fi
        progress=1
        break
      fi
    done
    [[ "${progress}" -eq 1 ]] || sleep 2
  done
  return "${failures}"
}

run_wave() {
  case "${SCHEDULER_MODE}" in
    round_robin)
      run_wave_round_robin "$@"
      ;;
    per_gpu)
      if [[ "${PROCESSES_PER_GPU}" -lt 1 ]]; then
        echo "PROCESSES_PER_GPU must be >= 1" >&2
        return 2
      fi
      run_wave_per_gpu "$@"
      ;;
    *)
      echo "Unknown SCHEDULER_MODE: ${SCHEDULER_MODE}" >&2
      return 2
      ;;
  esac
}

jobs=()
tail_jobs_small=()
tail_jobs_16=()
tail_jobs_32=()
case "${STAGE}" in
  all|unified)
    echo "DFlex blocked: no tracked formal DFlex config/provenance; excluded from unified queue." >&2
    for model_type in base full; do
      jobs+=( $'results/formal_671_core\t'"${model_type}"$'/seed_42\tconf/config_formal_671_'"${model_type}"$'.yaml\t--model-type '"${model_type}"$' --alpha 0.0 --seed 42 --nmul 1' )
    done
    for lambda in 0003 0005 0007 0010 0015 0020 0030 0050 0100; do
      case "${lambda}" in
        0003) alpha=0.003 ;; 0005) alpha=0.005 ;; 0007) alpha=0.007 ;;
        0010) alpha=0.010 ;; 0015) alpha=0.015 ;; 0020) alpha=0.020 ;;
        0030) alpha=0.030 ;; 0050) alpha=0.050 ;; 0100) alpha=0.100 ;;
      esac
      jobs+=( $'results/formal_671_core\tlambda'"${lambda}"$'/seed_42\tconf/config_formal_671_flex_lambda'"${lambda}"$'.yaml\t--model-type flex --alpha '"${alpha}"$' --seed 42 --nmul 1' )
    done
    for seed in 43 44; do
      jobs+=( $'results/formal_671_reference\tflex_lambda0007/seed_'"${seed}"$'\tconf/config_formal_671_flex_lambda0007_seed'"${seed}"$'.yaml\t--model-type flex --alpha 0.007 --seed '"${seed}"$' --nmul 1' )
    done
    for process in w_snow w_sub w_phen w_int; do
      for seed in 42 43 44; do
        jobs+=( $'results/formal_671_lopo\t'"${process}"$'/seed_'"${seed}"$'\tconf/config_formal_671_loro.yaml\t--model-type flex --alpha 0.007 --seed '"${seed}"$' --nmul 1 --removed-process '"${process}" )
      done
    done
    for nmul in 1 4 8; do
      for seed in 42 43 44; do
        tail_jobs_small+=( "results/formal_671_nmul"$'\t'"lambda0007/nmul${nmul}/seed_${seed}"$'\t'"conf/config_formal_671_flex_lambda0007.yaml"$'\t'"--model-type flex --alpha 0.007 --nmul ${nmul} --seed ${seed}" )
      done
    done
    for seed in 42 43 44; do
      tail_jobs_16+=( "results/formal_671_nmul"$'\t'"lambda0007/nmul16/seed_${seed}"$'\t'"conf/config_formal_671_flex_lambda0007.yaml"$'\t'"--model-type flex --alpha 0.007 --nmul 16 --seed ${seed}" )
      tail_jobs_32+=( "results/formal_671_nmul"$'\t'"lambda0007/nmul32/seed_${seed}"$'\t'"conf/config_formal_671_flex_lambda0007.yaml"$'\t'"--model-type flex --alpha 0.007 --nmul 32 --seed ${seed}" )
    done
    for region in 0 1 2 3 4 5 6; do
      for model_type in base full flex; do
        for seed in 42 43 44; do
          alpha="0.0"; [[ "${model_type}" == "flex" ]] && alpha="0.007"
          jobs+=( $'results/formal_671_loro\t'"${model_type}"$'/region'"${region}"$'/seed_'"${seed}"$'\tconf/config_formal_671_loro.yaml\t--model-type '"${model_type}"$' --alpha '"${alpha}"$' --seed '"${seed}"$' --nmul 1 --loro-holdout-region '"${region}" )
        done
      done
    done
    ;;
  reference)
    for seed in 43 44; do
      jobs+=( $'results/formal_671_reference\tflex_lambda0007/seed_'"${seed}"$'\tconf/config_formal_671_flex_lambda0007_seed'"${seed}"$'.yaml\t--model-type flex --alpha 0.007 --seed '"${seed}"$' --nmul 1' )
    done
    ;;
  lopo)
    for process in w_snow w_sub w_phen w_int; do
      for seed in 42 43 44; do
        jobs+=( $'results/formal_671_lopo\t'"${process}"$'/seed_'"${seed}"$'\tconf/config_formal_671_loro.yaml\t--model-type flex --alpha 0.007 --seed '"${seed}"$' --nmul 1 --removed-process '"${process}" )
      done
    done
    ;;
  nmul)
    for nmul in 1 8 16 32; do
      for seed in 42 43 44; do
        jobs+=( $'results/formal_671_nmul\tlambda0007/nmul'"${nmul}"$'/seed_'"${seed}"$'\tconf/config_formal_671_flex_lambda0007.yaml\t--model-type flex --alpha 0.007 --nmul '"${nmul}"' --seed '"${seed}" )
      done
    done
    ;;
  loro)
    for region in 0 1 2 3 4 5 6; do
      for model_type in base full flex; do
        for seed in 42 43 44; do
          alpha="0.0"; [[ "${model_type}" == "flex" ]] && alpha="0.007"
          jobs+=( "results/formal_671_loro"$'\t'"${model_type}/region${region}/seed_${seed}"$'\t'"conf/config_formal_671_loro.yaml"$'\t'"--model-type ${model_type} --alpha ${alpha} --seed ${seed} --nmul 1 --loro-holdout-region ${region}" )
        done
      done
    done
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac

tail_jobs_count=$(( ${#tail_jobs_small[@]} + ${#tail_jobs_16[@]} + ${#tail_jobs_32[@]} ))
echo "stage=${STAGE} main_jobs=${#jobs[@]} tail_jobs=${tail_jobs_count} gpus=${GPU_IDS_STR} scheduler=${SCHEDULER_MODE} processes_per_gpu=${PROCESSES_PER_GPU} max_parallel=${MAX_PARALLEL} patience=${PATIENCE}"
if [[ "${STAGE}" == "unified" && "${tail_jobs_count}" -gt 0 ]]; then
  small_slots="${NMUL_SMALL_PROCESSES_PER_GPU}"
  nmul16_slots="${NMUL16_PROCESSES_PER_GPU}"
  nmul32_slots="${NMUL32_PROCESSES_PER_GPU}"
  main_failures=0
  tail_failures=0
  echo "phase=main non_sensitivity_nmul=1 processes_per_gpu=${PROCESSES_PER_GPU} jobs=${#jobs[@]}"
  if ! run_wave jobs; then
    main_failures=1
  fi
  echo "phase=nmul_tail_small values=1,4,8 processes_per_gpu=${small_slots} jobs=${#tail_jobs_small[@]}"
  if ! PROCESSES_PER_GPU="${small_slots}" run_wave tail_jobs_small; then
    tail_failures=1
  fi
  echo "phase=nmul_tail_16 values=16 processes_per_gpu=${nmul16_slots} jobs=${#tail_jobs_16[@]}"
  if ! PROCESSES_PER_GPU="${nmul16_slots}" run_wave tail_jobs_16; then
    tail_failures=1
  fi
  echo "phase=nmul_tail_32 values=32 processes_per_gpu=${nmul32_slots} jobs=${#tail_jobs_32[@]}"
  if ! PROCESSES_PER_GPU="${nmul32_slots}" run_wave tail_jobs_32; then
    tail_failures=1
  fi
  if [[ "${main_failures}" -ne 0 || "${tail_failures}" -ne 0 ]]; then
    exit 1
  fi
else
  run_wave jobs
fi
echo "STAGE COMPLETE: ${STAGE}"
