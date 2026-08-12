#!/bin/bash
# Block 1 Full-MOPEX LOPO ablation: 4 ablations x 3 seeds.
# Runs at most two ablation Python processes in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

GPU_IDS_STR=${1:-"0"}
MAX_PARALLEL=${2:-2}
PYTHON_BIN="${PYTHON_BIN:-/home/jingxin/code/dmg-research/.venv/bin/python}"
CONFIG="${CONFIG:-conf/config_dmopex_v1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/block1_full_lopo}"
LOG_DIR="${LOG_DIR:-logs/block1_full_lopo}"
ALPHA="${ALPHA:-0.0}"
SEEDS=(${SEEDS:-42 123 456})

IFS=',' read -ra GPUS <<< "$GPU_IDS_STR"
NUM_GPUS=${#GPUS[@]}

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
export FLEXMOPEX_DATA_DIR="${FLEXMOPEX_DATA_DIR:-$PROJECT_DIR/../../data}"
export DATA_PATH="${DATA_PATH:-$PROJECT_DIR/../../data}"
export BASIN_GROUPS_DIR="${BASIN_GROUPS_DIR:-$PROJECT_DIR/../../data/basin_groups}"
export GAGE_INFO="${GAGE_INFO:-$PROJECT_DIR/../../data/gage_id.npy}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_inductor_cache}"
export TORCH_COMPILE_DEBUG=0
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

ABLATION_NAMES=(
  "full_minus_phenology"
  "full_minus_interception"
  "full_minus_snow"
  "full_minus_subsurface"
)
ABLATION_WEIGHTS=(
  "0 1 1 1"
  "1 0 1 1"
  "1 1 0 1"
  "1 1 1 0"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

is_complete() {
  local ablation_name=$1
  local seed=$2
  compgen -G "$OUTPUT_ROOT/config_dmopex_v1/${ablation_name}/seed_${seed}/test*_Ep*/metrics_agg.json" > /dev/null
}

run_ablation() {
  local ablation_name=$1
  local weights=$2
  local gpu=$3
  local log_file="$LOG_DIR/${ablation_name}.log"
  local missing=0
  for seed in "${SEEDS[@]}"; do
    if ! is_complete "$ablation_name" "$seed"; then
      missing=1
      break
    fi
  done
  if [[ "$missing" -eq 0 ]]; then
    log "SKIP ${ablation_name} (all seeds complete)"
    return 0
  fi

  log "START gpu=${gpu} ${ablation_name} weights=(${weights})"
  "$PYTHON_BIN" run_batch_fixed_weights.py \
    --ablation-name "$ablation_name" \
    --fixed-weights ${weights} \
    --seeds "${SEEDS[@]}" \
    --gpu-id "$gpu" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --alpha "$ALPHA" \
    > "$log_file" 2>&1
  log "DONE  gpu=${gpu} ${ablation_name}"
}
export -f log is_complete run_ablation
export PYTHON_BIN CONFIG OUTPUT_ROOT LOG_DIR ALPHA TORCHINDUCTOR_CACHE_DIR TORCH_COMPILE_DEBUG
export FLEXMOPEX_DATA_DIR DATA_PATH BASIN_GROUPS_DIR GAGE_INFO

PIDS=()
JOB_IDX=0

for idx in "${!ABLATION_NAMES[@]}"; do
  ablation_name="${ABLATION_NAMES[$idx]}"
  weights="${ABLATION_WEIGHTS[$idx]}"
  gpu="${GPUS[$((JOB_IDX % NUM_GPUS))]}"

  while [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]]; do
    NEW_PIDS=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        NEW_PIDS+=("${pid}")
      fi
    done
    PIDS=("${NEW_PIDS[@]}")
    [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]] && sleep 5
  done

  run_ablation "${ablation_name}" "${weights}" "${gpu}" &
  PIDS+=($!)
  ((JOB_IDX++)) || true
done

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAILED=1
  fi
done

log "Collecting LOPO summaries"
"$PYTHON_BIN" scripts/collect_full_lopo_ablation.py --root "$OUTPUT_ROOT" --seeds "${SEEDS[@]}"

if [[ "$FAILED" -ne 0 ]]; then
  log "DONE with failed runs. Check ${LOG_DIR}/"
  exit 1
fi

log "DONE. Results in ${OUTPUT_ROOT}/analysis"
