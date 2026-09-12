#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${ROOT}/manuscript/supplement"
PYTHON_BIN="${ROOT}/../../.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
exec "${PYTHON_BIN}" "${OUT}/scripts/audit_s2_harness.py" --project-root "${ROOT}" --output-dir "${OUT}"
