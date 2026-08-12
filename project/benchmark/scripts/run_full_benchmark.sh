#!/usr/bin/env bash
# Consolidated Master Production Script for 36 Hydrological Models Benchmark & Evaluation
set -euo pipefail

BENCHMARK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ID=${1:-full300_production_frozen}
PY=${PYTHON:-/root/miniconda3/bin/python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${BENCHMARK_ROOT}:${BENCHMARK_ROOT}/src:${PYTHONPATH:-}"

echo "=========================================================="
echo "  Starting Consolidated 36-Model Hydrological Benchmark"
echo "  Run ID: ${RUN_ID}"
echo "  Root: ${BENCHMARK_ROOT}"
echo "=========================================================="

# Step 1: Execute CMA-ES Benchmark Training for all 36 models
"$PY" "${BENCHMARK_ROOT}/scripts/run_36model_benchmark.py" \
  --model all \
  --run-id "${RUN_ID}" \
  --config "${BENCHMARK_ROOT}/configs/full_run_10starts_300gen_warm1980_1981x5.yaml"

# Step 2: Perform Evaluation on Train & Test sets
"$PY" "${BENCHMARK_ROOT}/scripts/evaluate_benchmark_metrics.py" \
  --checkpoint-root "${BENCHMARK_ROOT}/checkpoints/${RUN_ID}" \
  --config "${BENCHMARK_ROOT}/configs/full_run_10starts_300gen_warm1980_1981x5.yaml" \
  --output-dir "${BENCHMARK_ROOT}/results"

echo "=========================================================="
echo "  Benchmark Run & Evaluation Completed Successfully!"
echo "  Results saved in: ${BENCHMARK_ROOT}/results"
echo "=========================================================="
