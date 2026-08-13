#!/usr/bin/env bash
# Canonical Full300 CMA-ES continuation / resume launcher (local).
#
# Trains only models that do NOT have a DONE marker under
# checkpoints/<RUN_ID>/<model>/, resuming each from its latest saved
# generation.  This is the continuation equivalent of the remote
# run_full300_remaining_continuation.sh workflow.
#
# Usage:
#   bash scripts/run_continuation.sh <RUN_ID> [MODEL ...]
#   bash scripts/run_continuation.sh full300_20260729_160112          # all unfinished
#   bash scripts/run_continuation.sh my_run mopex1 mopex2             # specific models
set -u -o pipefail

BENCHMARK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ID=${1:?usage: run_continuation.sh RUN_ID [MODEL ...]}
shift || true
PY=${PYTHON:-"${BENCHMARK_ROOT}/../.venv/bin/python"}
CONFIG=${CONFIG:-"${BENCHMARK_ROOT}/configs/full_run_10starts_300gen_warm1980_1981x5.yaml"}
CKPT_ROOT="${BENCHMARK_ROOT}/checkpoints/${RUN_ID}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${BENCHMARK_ROOT}:${BENCHMARK_ROOT}/src:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${BENCHMARK_ROOT}/logs/torchinductor_cache/${RUN_ID}"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=($(cd "${BENCHMARK_ROOT}" && "${PY}" - <<'EOF'
import sys
sys.path[:0] = ["", "src"]
from src.model_registry import NPARAM_INFO_36
print(" ".join(NPARAM_INFO_36.keys()))
EOF
))
fi

echo "=== Continuation run [$RUN_ID]: ${#MODELS[@]} model(s) ==="
mkdir -p "${CKPT_ROOT}" "${BENCHMARK_ROOT}/logs"
for model in "${MODELS[@]}"; do
  done_marker="${CKPT_ROOT}/${model}/DONE"
  if [ -f "${done_marker}" ]; then
    echo "[skip] ${model}: already completed"
    continue
  fi
  echo "=== [$(date -Is)] training ${model} (resume if checkpoints exist) ==="
  "$PY" "${BENCHMARK_ROOT}/scripts/run_36model_benchmark.py" \
    --model "${model}" \
    --run-id "${RUN_ID}" \
    --config "${CONFIG}" \
    --device cuda
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[failed] ${model} (rc=$rc)"
  else
    echo "[done] ${model}"
  fi
done
echo "=== Continuation finished ==="
