#!/usr/bin/env bash
# DPL-aligned IC Full300 training and evaluation.
set -euo pipefail

BENCHMARK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ID=${1:-ic_dpl_aligned_full300_20260819}
PY=${PYTHON:-/root/miniconda3/bin/python}
CONFIG=${CONFIG:-${BENCHMARK_ROOT}/configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-${BENCHMARK_ROOT}/results/${RUN_ID}_evaluation}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PYTHONPATH="${BENCHMARK_ROOT}:${BENCHMARK_ROOT}/src:${BENCHMARK_ROOT}/../..:${PYTHONPATH:-}"

"$PY" "${BENCHMARK_ROOT}/scripts/validate_full300_config.py" \
  --config "$CONFIG" \
  --manifest "${BENCHMARK_ROOT}/frozen_versions/cmaes36_dpl_aligned_20260819/manifest.json"

"$PY" "${BENCHMARK_ROOT}/scripts/run_36model_benchmark.py" \
  --model all \
  --run-id "$RUN_ID" \
  --config "$CONFIG"

"$PY" "${BENCHMARK_ROOT}/scripts/evaluate_benchmark_metrics.py" \
  --checkpoint-root "${BENCHMARK_ROOT}/checkpoints/${RUN_ID}" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR"
