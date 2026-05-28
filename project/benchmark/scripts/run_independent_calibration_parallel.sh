#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-project/benchmark/conf/benchmark.yaml}"
TASK_TABLE="${TASK_TABLE:-project/benchmark/outputs/tasks/independent_calibration_tasks.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-project/benchmark/outputs}"
LOG_DIR="${LOG_DIR:-project/benchmark/logs}"
JOBS="${JOBS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER_ARGS="${RUNNER_ARGS:-}"

usage() {
  printf 'Usage: %s [--config PATH] [--task-table PATH] [--output-dir DIR] [--log-dir DIR] [--jobs N]\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --task-table) TASK_TABLE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
export CONFIG OUTPUT_DIR LOG_DIR PYTHON_BIN RUNNER_ARGS

if [[ ! -f "$TASK_TABLE" ]]; then
  "$PYTHON_BIN" project/benchmark/generate_tasks.py \
    --config "$CONFIG" \
    --output "$TASK_TABLE"
fi

tail -n +2 "$TASK_TABLE" | xargs -P "$JOBS" -I{} bash -c '
  IFS=, read -r basin_id model_id objective <<< "$1"
  log_file="${LOG_DIR}/${basin_id}_${model_id}_${objective}.log"
  "${PYTHON_BIN}" project/benchmark/run_independent_calibration.py \
    --config "${CONFIG}" \
    --basin-id "${basin_id}" \
    --model-id "${model_id}" \
    --objective "${objective}" \
    --output-dir "${OUTPUT_DIR}" \
    ${RUNNER_ARGS} >"${log_file}" 2>&1
' _ {}
