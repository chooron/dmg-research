#!/bin/bash
# Binary-Flex-MOPEX pilot: seed=42, 5 alphas, max 2 parallel jobs.
# Run from project/flexmopex/:
#   bash scripts/run_binary_pilot.sh
#
# Staged execution:
#   Step 1 — preflight (no training)
#   Step 2 — 2-epoch smoke test (alpha=0.005)
#   Step 3 — single full run (alpha=0.005)
#   Step 4 — remaining alphas (0.001, 0.003, 0.01, 0.03)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MAX_PARALLEL=2
SEED=42
CONFIG="conf/config_binary_v1.yaml"
OUTPUT_ROOT="outputs/binary_pilot"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_slots() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 10
    done
}

# ── Step 1: Preflight ────────────────────────────────────────────────────────
log "Step 1: preflight check"
python run_model.py \
    --config "$CONFIG" \
    --model-type binary \
    --alpha 0.01 \
    --seed "$SEED" \
    --preflight-only
log "Preflight passed."

# ── Step 2: 2-epoch smoke test ───────────────────────────────────────────────
log "Step 2: 2-epoch smoke test (alpha=0.005)"
python run_model.py \
    --config "$CONFIG" \
    --model-type binary \
    --alpha 0.005 \
    --seed "$SEED" \
    --epochs 2 \
    --mode train \
    --output-root "$OUTPUT_ROOT" \
    --run-name "binary_smoke_alpha0.005_seed${SEED}"
log "Smoke test complete. Check outputs/${OUTPUT_ROOT}/binary_smoke_alpha0.005_seed${SEED}/sim/ for p_* and z_* files."

# ── Step 3: Single full run (alpha=0.005) ────────────────────────────────────
log "Step 3: full run alpha=0.005"
python run_model.py \
    --config "$CONFIG" \
    --model-type binary \
    --alpha 0.005 \
    --seed "$SEED" \
    --mode train_test \
    --output-root "$OUTPUT_ROOT" \
    --run-name "binary_alpha0.005_seed${SEED}"
log "Full run alpha=0.005 complete."

# ── Step 4: Remaining alphas in parallel ─────────────────────────────────────
log "Step 4: remaining alphas (0.001, 0.003, 0.01, 0.03)"
for alpha in 0.001 0.003 0.01 0.03; do
    wait_for_slots
    log "Launching alpha=${alpha}"
    python run_model.py \
        --config "$CONFIG" \
        --model-type binary \
        --alpha "$alpha" \
        --seed "$SEED" \
        --mode train_test \
        --output-root "$OUTPUT_ROOT" \
        --run-name "binary_alpha${alpha}_seed${SEED}" \
        >> "logs/binary_alpha${alpha}_seed${SEED}.log" 2>&1 &
done

wait
log "All pilot runs complete."
log "Run analysis: python analysis/run_binary_flex_pilot_analysis.py"
