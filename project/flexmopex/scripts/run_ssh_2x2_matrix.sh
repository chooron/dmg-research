#!/usr/bin/env bash
# Persistent Pure-X35 SSH 2x2 factorial launcher.
# One process per visible GPU; with one GPU this intentionally queues 12 runs.
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_DIR="$ROOT_DIR/project/flexmopex"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/ssh_2x2}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"
LOG_ROOT="$RESULT_ROOT/logs"
STATUS_FILE="$RESULT_ROOT/matrix_status.tsv"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/.cache/torch_inductor}"
export TORCH_COMPILE_DEBUG=0

printf 'condition\tseed\tgpu\tstatus\tpid\tconfig\toutput\tlog\tstarted\tended\n' > "$STATUS_FILE"

"$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable; refusing to run on CPU"
assert hasattr(torch, "compile"), "torch.compile is unavailable; refusing to run"
print(f"GPU={torch.cuda.get_device_name(0)} count={torch.cuda.device_count()} torch={torch.__version__}", flush=True)
PY

# Paired seed blocks: E1/E2/E3/E4 for seed 42, then 43, then 44.
for seed in 42 43 44; do
  for condition in E1 E2 E3 E4; do
    config="$PROJECT_DIR/conf/ssh_2x2/config_${condition}_pure_x35_531_lambda0007.yaml"
    output="$RESULT_ROOT/$condition/seed_$seed"
    log="$LOG_ROOT/${condition}_seed${seed}.log"
    started="$(date -Is)"
    printf '%s\t%s\t%s\tRUNNING\t%s\t%s\t%s\t%s\t%s\t\n' \
      "$condition" "$seed" "$GPU_ID" "$$" "$config" "$output" "$log" "$started" >> "$STATUS_FILE"
    echo "[matrix] START condition=$condition seed=$seed gpu=$GPU_ID pid=$$ config=$config output=$output log=$log" | tee -a "$LOG_ROOT/matrix.log"

    set +e
    "$PYTHON_BIN" "$PROJECT_DIR/run_model.py" \
      --config "$config" \
      --mode train_test \
      --seed "$seed" \
      --gpu-id "$GPU_ID" \
      --epochs 10 \
      --test-epoch 10 \
      --disable-early-stopping \
      --output-root "$RESULT_ROOT" \
      --run-name "$condition/seed_$seed" \
      --verbose > "$log" 2>&1
    rc=$?
    set -e
    ended="$(date -Is)"
    if [[ "$rc" -eq 0 ]]; then status=COMPLETED; else status=FAILED; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$condition" "$seed" "$GPU_ID" "$status" "$$" "$config" "$output" "$log" "$started" "$ended" >> "$STATUS_FILE"
    echo "[matrix] END condition=$condition seed=$seed status=$status rc=$rc" | tee -a "$LOG_ROOT/matrix.log"
  done
done

echo "[matrix] ALL QUEUED RUNS FINISHED: $STATUS_FILE" | tee -a "$LOG_ROOT/matrix.log"
