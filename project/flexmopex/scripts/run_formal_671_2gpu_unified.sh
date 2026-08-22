#!/usr/bin/env bash
# Unified CAMELS-671 formal launcher for a two-GPU host.
# Keeps six training slots on each GPU (12 total) and replenishes a shared queue
# as runs finish; reference/LOPO/nmul/LORO jobs are not run in category waves.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"

GPU_IDS="${GPU_IDS:-0,1}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-6}"
NMUL_PROCESSES_PER_GPU="${NMUL_PROCESSES_PER_GPU:-3}"
PATIENCE="${PATIENCE:-10}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
MIN_EPOCHS="${MIN_EPOCHS:-50}"
MIN_DELTA="${MIN_DELTA:-0.0001}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-results/formal_671_unified_nmul1_tail3}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"

if [[ "${PYTHON_BIN}" == "python" ]] && ! command -v python >/dev/null 2>&1 && [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif [[ "${PYTHON_BIN}" != /* && "${PYTHON_BIN}" == */* && ! -x "${PYTHON_BIN}" ]]; then
  if [[ -x "${PROJECT_DIR}/${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/${PYTHON_BIN}"
  elif [[ -x "${REPO_ROOT}/${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${REPO_ROOT}/${PYTHON_BIN}"
  fi
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne 2 || -z "${GPUS[0]}" || -z "${GPUS[1]}" ]]; then
  echo "GPU_IDS must name exactly two GPUs, for example GPU_IDS=0,1" >&2
  exit 2
fi
if [[ "${GPUS[0]}" == "${GPUS[1]}" ]]; then
  echo "GPU_IDS must contain two distinct GPU ids" >&2
  exit 2
fi
if [[ "${PROCESSES_PER_GPU}" -ne 6 ]]; then
  echo "PROCESSES_PER_GPU is fixed at 6 for the formal two-GPU protocol" >&2
  exit 2
fi
if [[ "${NMUL_PROCESSES_PER_GPU}" -ne 3 ]]; then
  echo "NMUL_PROCESSES_PER_GPU is fixed at 3 for the nmul sensitivity tail" >&2
  exit 2
fi
if [[ "${PATIENCE}" -ne 10 ]]; then
  echo "PATIENCE is fixed at 10 for the formal 671 protocol" >&2
  exit 2
fi
if [[ "${MAX_EPOCHS}" -ne 100 || "${MIN_EPOCHS}" -ne 50 ]]; then
  echo "MAX_EPOCHS/MIN_EPOCHS are fixed at 100/50 for the formal 671 protocol" >&2
  exit 2
fi
if [[ "${FORCE_OVERWRITE:-0}" == "1" ]]; then
  echo "FORCE_OVERWRITE=1 is not allowed by the formal unified launcher" >&2
  exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

BASIN_MANIFEST="${REPO_ROOT}/data/gage_id.npy"
if [[ ! -f "${BASIN_MANIFEST}" ]]; then
  echo "Missing 671-basin manifest: ${BASIN_MANIFEST}" >&2
  exit 2
fi
BASIN_COUNT="$(${PYTHON_BIN} - "${BASIN_MANIFEST}" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
values = np.load(path, allow_pickle=False)
print(values.shape[0])
PY
)"
if [[ "${BASIN_COUNT}" != "671" ]]; then
  echo "Refusing to launch: ${BASIN_MANIFEST} contains ${BASIN_COUNT} basins, expected 671" >&2
  exit 2
fi

CONFIG_COUNT=0
for config in "${PROJECT_DIR}"/conf/config_formal_671_*.yaml; do
  [[ -f "${config}" ]] || continue
  CONFIG_COUNT=$((CONFIG_COUNT + 1))
  grep -q '^basin_count: 671$' "${config}" || {
    echo "Formal config is not marked basin_count=671: ${config}" >&2
    exit 2
  }
  grep -q '^basin_manifest: data/gage_id.npy$' "${config}" || {
    echo "Formal config is not bound to data/gage_id.npy: ${config}" >&2
    exit 2
  }
done
if [[ "${CONFIG_COUNT}" -eq 0 ]]; then
  echo "No config_formal_671_*.yaml files found" >&2
  exit 2
fi

export GPU_IDS
export PROCESSES_PER_GPU
export NMUL_PROCESSES_PER_GPU
export SCHEDULER_MODE=per_gpu
export PATIENCE
export MAX_EPOCHS
export MIN_EPOCHS
export MIN_DELTA
export FORCE_OVERWRITE=0
export RESUME_INCOMPLETE
export MAX_PARALLEL=12
export PYTHON_BIN
export RESULTS_ROOT

cat <<EOF
Flex-MOPEX formal 671 unified launch
  GPUs: ${GPU_IDS}
  process slots: ${PROCESSES_PER_GPU} per GPU (${MAX_PARALLEL} total)
  nmul tail slots: ${NMUL_PROCESSES_PER_GPU} per GPU
  basin manifest: ${BASIN_MANIFEST} (${BASIN_COUNT})
  configs checked: ${CONFIG_COUNT}
  epochs: ${MAX_EPOCHS} (minimum ${MIN_EPOCHS})
  early-stop patience: ${PATIENCE}
  results root: ${RESULTS_ROOT}
  resume incomplete: ${RESUME_INCOMPLETE}
EOF

exec bash "${SCRIPT_DIR}/run_formal_671_staged.sh" unified
