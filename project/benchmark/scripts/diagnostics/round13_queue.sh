#!/usr/bin/env bash
set -u

ROOT="/home/jingxin/code/dmg-research/project/benchmark"
STATUS="$ROOT/results/dpl_round13_20260805/auto100/status.csv"
LOGDIR="/tmp/round13_auto_queue"
mkdir -p "$LOGDIR"

completed() {
    awk -F, -v m="$1" '$1 == m && $3 == "COMPLETED" { found = 1 } END { exit(found ? 0 : 1) }' "$STATUS" 2>/dev/null
}

running() {
    pgrep -f "round13_m1.py --arm auto100 --models $1" >/dev/null 2>&1
}

start_model() {
    local model="$1"
    if completed "$model" || running "$model"; then
        return
    fi
    nohup python scripts/diagnostics/round13_m1.py --arm auto100 --models "$model" \
        >"$LOGDIR/${model}.log" 2>&1 &
}

MODELS=(susannah1 susannah2 tank tcm topmodel us1 vic wetland xinanjiang)

while true; do
    active=0
    unfinished=0
    for model in "${MODELS[@]}"; do
        if ! completed "$model"; then
            unfinished=1
        fi
        if running "$model"; then
            active=$((active + 1))
        fi
    done

    [ "$unfinished" -eq 0 ] && break

    for model in "${MODELS[@]}"; do
        [ "$active" -ge 3 ] && break
        if ! completed "$model" && ! running "$model"; then
            start_model "$model"
            active=$((active + 1))
        fi
    done
    sleep 20
done

touch "$LOGDIR/COMPLETE"
