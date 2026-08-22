#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_DIR="$ROOT_DIR/project/flexmopex"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/ssh_2x2}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"
LOG_ROOT="$RESULT_ROOT/logs"
STATUS_FILE="$RESULT_ROOT/matrix_status.tsv"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
export FLEXMOPEX_DATA_DIR="${FLEXMOPEX_DATA_DIR:-/root/autodl-fs/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/.cache/torch_inductor}"
export TORCH_COMPILE_DEBUG=0
printf 'condition\tseed\tgpu\tstatus\tpid\tconfig\toutput\tlog\tstarted\tended\n' > "$STATUS_FILE"
for seed in 42 43 44; do
  pids=()
  conditions=(E1 E2 E3 E4)
  for condition in "${conditions[@]}"; do
    config="$PROJECT_DIR/conf/ssh_2x2/config_${condition}_pure_x35_531_lambda0007.yaml"
    output="$RESULT_ROOT/$condition/seed_$seed"
    log="$LOG_ROOT/${condition}_seed${seed}.log"
    started="$(date -Is)"
    mkdir -p "$output"
    printf '%s\t%s\t%s\tRUNNING\t%s\t%s\t%s\t%s\t%s\t\n' "$condition" "$seed" "$GPU_ID" "pending" "$config" "$output" "$log" "$started" >> "$STATUS_FILE"
    echo "[parallel4] START condition=$condition seed=$seed gpu=$GPU_ID config=$config output=$output log=$log" | tee -a "$LOG_ROOT/matrix.log"
    "$PYTHON_BIN" "$PROJECT_DIR/run_model.py" --config "$config" --mode train_test --seed "$seed" --gpu-id "$GPU_ID" --epochs 10 --test-epoch 10 --disable-early-stopping --output-root "$RESULT_ROOT" --run-name "$condition/seed_$seed" --verbose > "$log" 2>&1 &
    pids+=("$!")
  done
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"; condition="${conditions[$i]}"
    if wait "$pid"; then status=COMPLETED; rc=0; else status=FAILED; rc=$?; fi
    ended="$(date -Is)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t%s\n' "$condition" "$seed" "$GPU_ID" "$status" "$pid" "$PROJECT_DIR/conf/ssh_2x2/config_${condition}_pure_x35_531_lambda0007.yaml" "$RESULT_ROOT/$condition/seed_$seed" "$LOG_ROOT/${condition}_seed${seed}.log" "$ended" >> "$STATUS_FILE"
    echo "[parallel4] END condition=$condition seed=$seed pid=$pid status=$status rc=$rc" | tee -a "$LOG_ROOT/matrix.log"
  done
done
echo '[parallel4] ALL 12 RUNS FINISHED' | tee -a "$LOG_ROOT/matrix.log"
