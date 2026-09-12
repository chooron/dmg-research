#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SUPP="$ROOT/manuscript/supplement"
PY="$ROOT/../../.venv/bin/python"
if [ ! -x "$PY" ]; then PY="${PYTHON:-python3}"; fi
mkdir -p "$SUPP/logs" "$SUPP/results"
LOG="$SUPP/logs/s2_existing_validation_run.log"
exec > >(tee "$LOG") 2>&1
set -x
"$PY" "$SUPP/scripts/discover_existing_model_validations.py" --project-root "$ROOT" --output-dir "$SUPP"
"$PY" -m pytest -p no:cacheprovider -q \
  "$ROOT/tests/test_models_grad.py" \
  "$ROOT/tests/test_gr4j_x4_gradient.py" \
  "$ROOT/tests/test_composed_temperature_delay.py" \
  "$ROOT/tests/test_simhyd.py" \
  "$ROOT/tests/test_xaj_fused_compositions.py" \
  --junitxml="$SUPP/results/s2_existing_pytest.xml"
