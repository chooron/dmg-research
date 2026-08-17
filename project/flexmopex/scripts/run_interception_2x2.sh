#!/bin/bash
# Interception 2x2 experiment — fresh matched run (Phases 4-7).
#
#   Arm A: V0-original        (production shared PET, original w_int x alpha)
#   Arm B: V0-decoupled       (production shared PET, amplitude-decoupled)
#   Arm C: V1-original        (independent interception-loss PET, original)
#   Arm D: V1-decoupled       (independent interception-loss PET, decoupled)
#
# All arms: aic_alpha=0.01, seed=42, 10 epochs, camels_671, canonical
# split/warmup/optimizer/lr/batch, same architecture/AIC table/trainable
# output count, identical initial weights (shared epoch-0 state from seed 42).
#
# Usage: bash scripts/run_interception_2x2.sh [GPU_ID] [PYTHON]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU_ID=${1:-0}
PY=${2:-"$(cd ../.. && pwd)/.venv/bin/python"}
if [ ! -x "$PY" ]; then PY=python; fi
ROOT="results/intercept_2x2"

# ---------------------------------------------------------------------------
# Phase 5 — matched initialization: one shared epoch-0 state (arm A, seed 42)
# ---------------------------------------------------------------------------
echo "=== [init] building shared epoch-0 state (seed 42) ==="
$PY scripts/init_interception_2x2.py \
    --config conf/config_dmopex_intercept2x2_A.yaml \
    --output-root "$ROOT" --run-name A --gpu-id "$GPU_ID"

for ARM in B C D; do
    mkdir -p "$ROOT/$ARM/model"
    # rename to each arm's own checkpoint naming (model class name differs)
    case $ARM in
        B) NAME="learnedweightmopexdecoupled";;
        C) NAME="learnedweightmopexv1";;
        D) NAME="learnedweightmopexv1decoupled";;
    esac
    cp "$ROOT/A/model/learnedweightmopex_ep0.pt" "$ROOT/$ARM/model/${NAME}_ep0.pt"
    cp "$ROOT/A/model/trainer_state_ep0.pt"      "$ROOT/$ARM/model/trainer_state_ep0.pt"
done

echo "=== [init] initial-state equality (sha256 of the four ep0 model files) ==="
sha256sum "$ROOT"/*/model/*_ep0.pt

# ---------------------------------------------------------------------------
# Phase 4+6 — fresh 10-epoch runs (resume from ep0 => start_epoch 1) + canonical
# test evaluation (1995-2010) at epoch 10.  Never touches canonical results.
# ---------------------------------------------------------------------------
for ARM in A B C D; do
    if [ -f "$ROOT/$ARM/sim/metrics_agg.json" ]; then
        echo "=== [train] arm $ARM already complete (metrics_agg.json exists), skipping ==="
        continue
    fi
    echo "=== [train] arm $ARM (10 epochs, aic_alpha=0.01, seed=42) ==="
    $PY run_model.py \
        --config "conf/config_dmopex_intercept2x2_${ARM}.yaml" \
        --mode train_test \
        --epochs 10 \
        --gpu-id "$GPU_ID" \
        --output-root "$ROOT" \
        --run-name "$ARM" \
        > "$ROOT/$ARM/train.log" 2>&1
    echo "=== [train] arm $ARM done (log: $ROOT/$ARM/train.log) ==="
done

echo ""
echo "Interception 2x2 training complete. Results in $ROOT/"
echo "Next: python scripts/analyze_interception_2x2.py --output-root $ROOT"
