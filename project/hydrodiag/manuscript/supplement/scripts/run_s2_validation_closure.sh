#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SUPP="$(cd "$HERE/.." && pwd)"
PROJECT="$(cd "$SUPP/../.." && pwd)"
REPO="$(cd "$PROJECT/../.." && pwd)"
PY="$REPO/.venv/bin/python"
if [ ! -x "$PY" ]; then PY="\${PYTHON:-python3}"; fi
mkdir -p "$SUPP/results" "$SUPP/logs" "$SUPP/reports"
LOG="$SUPP/results/s2_validation_run.log"
exec > >(tee "$LOG") 2>&1
set -x
date -Is
echo "PROJECT=$PROJECT"
echo "REPO=$REPO"
bash "$HERE/run_existing_model_validations.sh" || true
"$PY" "$HERE/verify_s2_mass_balance.py" --project-root "$PROJECT" || true
"$PY" "$HERE/verify_s2_gradients.py" --project-root "$PROJECT" || true
"$PY" "$HERE/verify_s2_reference_equivalence.py" --project-root "$PROJECT" || true
"$PY" "$HERE/finalize_s2_validation_outputs.py" --supplement "$SUPP"
date -Is
