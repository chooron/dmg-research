#!/usr/bin/env bash
# Staged CAMELS-531 formal runner. No result directory is ever overwritten.
# Usage: GPU_IDS=0,1 bash scripts/run_formal_531_staged.sh reference|lopo|nmul|loro|dflex
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_STR="${GPU_IDS:-0}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS_STR}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPUS[@]}}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
MIN_EPOCHS="${MIN_EPOCHS:-50}"
PATIENCE="${PATIENCE:-20}"
MIN_DELTA="${MIN_DELTA:-0.0001}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
STAGE="${1:-}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
if [[ -z "${STAGE}" ]]; then
  echo "Usage: GPU_IDS=0,1 $0 reference|lopo|dflex|nmul|loro" >&2
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
  local run_dir="${output_root}/${run_name}"
  mkdir -p "${run_dir}"
  if [[ "${FORCE_OVERWRITE}" != "1" ]] && [[ -e "${run_dir}/model" || -e "${run_dir}/sim" || -e "${run_dir}/early_stopping.json" ]]; then
    if [[ -f "${run_dir}/early_stopping.json" && -f "${run_dir}/sim/streamflow.npy" ]]; then
      echo "SKIP complete ${run_name}"
      return 0
    fi
    echo "REFUSE non-empty incomplete output: ${run_dir}" >&2
    return 1
  fi
  echo "START stage=${STAGE} gpu=${gpu} run=${run_name}"
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
    "$@" > "${run_dir}/train.log" 2>&1
  echo "DONE stage=${STAGE} run=${run_name}"
}

run_wave() {
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

jobs=()
case "${STAGE}" in
  all|unified)
    echo "DFlex blocked: no tracked formal DFlex config/provenance; excluded from unified queue." >&2
    for seed in 43 44; do
      jobs+=( $'results/formal_531_reference\tflex_lambda0007/seed_'"${seed}"$'\tconf/config_formal_531_flex_lambda0007_seed'"${seed}"$'.yaml\t--model-type flex --alpha 0.007 --seed '"${seed}" )
    done
    for process in w_snow w_sub w_phen w_int; do
      for seed in 42 43 44; do
        jobs+=( $'results/formal_531_lopo\t'"${process}"$'/seed_'"${seed}"$'\tconf/config_formal_531_loro.yaml\t--model-type flex --alpha 0.007 --seed '"${seed}"$' --removed-process '"${process}" )
      done
    done
    for nmul in 1 8 16 32; do
      for seed in 42 43 44; do
        jobs+=( $'results/formal_531_nmul\tlambda0007/nmul'"${nmul}"$'/seed_'"${seed}"$'\tconf/config_formal_531_flex_lambda0007.yaml\t--model-type flex --alpha 0.007 --nmul '"${nmul}"$' --seed '"${seed}" )
      done
    done
    for region in 0 1 2 3 4 5 6; do
      for model_type in base full flex; do
        for seed in 42 43 44; do
          alpha="0.0"; [[ "${model_type}" == "flex" ]] && alpha="0.007"
          jobs+=( $'results/formal_531_loro\t'"${model_type}"$'/region'"${region}"$'/seed_'"${seed}"$'\tconf/config_formal_531_loro.yaml\t--model-type '"${model_type}"$' --alpha '"${alpha}"$' --seed '"${seed}"$' --loro-holdout-region '"${region}" )
        done
      done
    done
    ;;
  reference)
    for seed in 43 44; do
      jobs+=( $'results/formal_531_reference\tflex_lambda0007/seed_'"${seed}"$'\tconf/config_formal_531_flex_lambda0007_seed'"${seed}"$'.yaml\t--model-type flex --alpha 0.007 --seed '"${seed}" )
    done
    ;;
  lopo)
    for process in w_snow w_sub w_phen w_int; do
      for seed in 42 43 44; do
        jobs+=( $'results/formal_531_lopo\t'"${process}"$'/seed_'"${seed}"$'\tconf/config_formal_531_loro.yaml\t--model-type flex --alpha 0.007 --seed '"${seed}"' --removed-process '"${process}" )
      done
    done
    ;;
  nmul)
    for nmul in 1 8 16 32; do
      for seed in 42 43 44; do
        jobs+=( $'results/formal_531_nmul\tlambda0007/nmul'"${nmul}"$'/seed_'"${seed}"$'\tconf/config_formal_531_flex_lambda0007.yaml\t--model-type flex --alpha 0.007 --nmul '"${nmul}"' --seed '"${seed}" )
      done
    done
    ;;
  loro)
    for region in 0 1 2 3 4 5 6; do
      for model_type in base full flex; do
        for seed in 42 43 44; do
          alpha="0.0"; [[ "${model_type}" == "flex" ]] && alpha="0.007"
          jobs+=( $'results/formal_531_loro\t'"${model_type}"$'/region'"${region}"$'/seed_'"${seed}"$'\tconf/config_formal_531_loro.yaml\t--model-type '"${model_type}"' --alpha '"${alpha}"' --seed '"${seed}"' --loro-holdout-region '"${region}" )
        done
      done
    done
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac

echo "stage=${STAGE} jobs=${#jobs[@]} gpus=${GPU_IDS_STR} max_parallel=${MAX_PARALLEL}"
run_wave jobs
echo "STAGE COMPLETE: ${STAGE}"
