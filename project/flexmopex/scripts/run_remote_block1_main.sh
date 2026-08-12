#!/usr/bin/env bash
set -euo pipefail

# ── Remote connection settings ────────────────────────────────────────────────
HOST="${HOST:-connect.westb.seetacloud.com}"
PORT="${PORT:-15632}"
USER="${USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/dmg-research}"
REMOTE_FLEX="${REMOTE_ROOT}/project/flexmopex"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/miniconda3/bin/python}"
LOCAL_ROOT="${LOCAL_ROOT:-/workspace/autoresearch}"
LOCAL_ENV="${LOCAL_ENV:-${LOCAL_ROOT}/project/flexmopex/.env}"

# ── Experiment settings ───────────────────────────────────────────────────────
GPU="${GPU:-1}"
# GPU 0 is used by block3_loro; GPU 1 is free for block1_main
# 2× RTX 3080 Ti (12GB); batch_size=200 → ~8 parallel jobs
MAX_PARALLEL="${MAX_PARALLEL:-10}"
BATCH_SIZE="${BATCH_SIZE:-200}"
SEEDS="${SEEDS:-42 123 456 789 1024}"
# DRY_RUN=1 → 跳过 tmux，直接在前台跑第一个 job（用于调试）
DRY_RUN="${DRY_RUN:-0}"

SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=no)
REMOTE="${USER}@${HOST}"

# ── SSH/SCP auth ──────────────────────────────────────────────────────────────
if [[ -z "${REMOTE_PASS:-}" && -f "$LOCAL_ENV" ]]; then
  REMOTE_PASS="$(sed -n 's/^SSH_PASSWORD=//p' "$LOCAL_ENV" | tail -n 1)"
fi

# Fallback: hardcode password if not in .env (will be overridden by env var)
REMOTE_PASS="${REMOTE_PASS:-I1L2jZ+i0XKT}"

if command -v sshpass >/dev/null 2>&1 && [[ -n "${REMOTE_PASS:-}" ]]; then
  SSH=(sshpass -p "$REMOTE_PASS" ssh "${SSH_OPTS[@]}" "$REMOTE")
  SCP=(sshpass -p "$REMOTE_PASS" scp -P "$PORT" -o StrictHostKeyChecking=no)
elif [[ -n "${REMOTE_PASS:-}" ]]; then
  ASKPASS_FILE="$(mktemp)"
  cat > "$ASKPASS_FILE" <<'ASKPASS'
#!/usr/bin/env bash
printf "%s" "$OMX_SSH_PASSWORD"
ASKPASS
  chmod 700 "$ASKPASS_FILE"
  cleanup() { rm -f "$ASKPASS_FILE" "${WORKER_DIR:-}" 2>/dev/null || true; }
  trap cleanup EXIT
  SSH=(env SSH_ASKPASS="$ASKPASS_FILE" SSH_ASKPASS_REQUIRE=force DISPLAY=dummy OMX_SSH_PASSWORD="$REMOTE_PASS" setsid -w ssh "${SSH_OPTS[@]}" "$REMOTE")
  SCP=(env SSH_ASKPASS="$ASKPASS_FILE" SSH_ASKPASS_REQUIRE=force DISPLAY=dummy OMX_SSH_PASSWORD="$REMOTE_PASS" setsid -w scp -P "$PORT" -o StrictHostKeyChecking=no)
else
  SSH=(ssh "${SSH_OPTS[@]}" "$REMOTE")
  SCP=(scp -P "$PORT" -o StrictHostKeyChecking=no)
fi

# ── Per-window parallelism ────────────────────────────────────────────────────
PAR_A=$(( MAX_PARALLEL / 3 + 1 ))
PAR_B=$(( MAX_PARALLEL / 3 + 1 ))
PAR_C=$(( MAX_PARALLEL / 3 ))
[[ $PAR_C -lt 1 ]] && PAR_C=1

# ── Generate worker scripts locally then SCP them ────────────────────────────
WORKER_DIR="$(mktemp -d)"
trap 'rm -rf "$WORKER_DIR"' EXIT

make_worker() {
  local name=$1 par=$2
  shift 2
  local tasks=("$@")

  local out="$WORKER_DIR/${name}.sh"
  cat > "$out" <<HEADER
#!/usr/bin/env bash
set -euo pipefail
cd "\$(dirname "\$0")/.."
export PATH='/root/miniconda3/bin':\$PATH

GPU="${GPU}"
MAX_PARALLEL="${par}"
BATCH_SIZE="${BATCH_SIZE}"
OUT_ROOT="results/block1_main"

run_exp() {
  local model_type=\$1 alpha=\$2 seed=\$3
  local tag="\${model_type}_alpha\${alpha}_seed\${seed}"
  local save_path="\${OUT_ROOT}/\${model_type}/alpha\${alpha}/seed\${seed}"
  mkdir -p "\${save_path}"
  echo "[\$(date +%H:%M:%S)] START  gpu=\${GPU}  \${tag}"
  CUDA_VISIBLE_DEVICES=\${GPU} python run_model.py \\
    --model-type "\${model_type}" \\
    --alpha "\${alpha}" \\
    --seed "\${seed}" \\
    --batch-size "\${BATCH_SIZE}" \\
    --output-root "\${save_path}" \\
    > "\${save_path}/train.log" 2>&1
  echo "[\$(date +%H:%M:%S)] DONE   gpu=\${GPU}  \${tag}"
}

TASKS=(
HEADER

  for seed in $SEEDS; do
    for task_spec in "${tasks[@]}"; do
      echo "  \"${task_spec} ${seed}\"" >> "$out"
    done
  done

  cat >> "$out" <<'FOOTER'
)

PIDS=()
for task in "${TASKS[@]}"; do
  read -r model_type alpha seed <<< "${task}"
  while [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]]; do
    NEW_PIDS=()
    for pid in "${PIDS[@]}"; do
      kill -0 "${pid}" 2>/dev/null && NEW_PIDS+=("${pid}")
    done
    PIDS=("${NEW_PIDS[@]+"${NEW_PIDS[@]}"}")
    [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]] && sleep 3
  done
  run_exp "${model_type}" "${alpha}" "${seed}" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "${pid}" || echo "WARNING: job ${pid} failed"
done
echo "Window WINNAME complete."
FOOTER

  sed -i "s/Window WINNAME complete/Window ${name} complete/" "$out"
  chmod +x "$out"
}

# base_full: base + full (10 jobs × 5 seeds = 10 tasks)
make_worker "_worker_base_full" "$PAR_A" "base 0.0" "full 0.0"

# flex_low: flex 0.005 + flex 0.01 (10 tasks)
make_worker "_worker_flex_low"  "$PAR_B" "flex 0.005" "flex 0.01"

# flex_high: flex 0.03 (5 tasks)
make_worker "_worker_flex_high" "$PAR_C" "flex 0.03"

# ── Upload all files ──────────────────────────────────────────────────────────
FILES=(
  project/flexmopex/__init__.py
  project/flexmopex/run_model.py
  project/flexmopex/models/pub_sampler.py
  project/flexmopex/models/pub_trainer.py
  project/flexmopex/scripts/run_block1_main.sh
)

echo "Creating remote sync dir..."
"${SSH[@]}" "rm -rf /tmp/flexmopex_sync && mkdir -p /tmp/flexmopex_sync"

echo "Uploading code..."
for f in "${FILES[@]}"; do
  "${SCP[@]}" "${LOCAL_ROOT}/${f}" "${REMOTE}:/tmp/flexmopex_sync/$(basename "$f")"
done

echo "Uploading worker scripts..."
for w in _worker_base_full _worker_flex_low _worker_flex_high; do
  "${SCP[@]}" "${WORKER_DIR}/${w}.sh" "${REMOTE}:/tmp/flexmopex_sync/${w}.sh"
done

# ── Install on remote ─────────────────────────────────────────────────────────
SESSION="block1_main"

echo "Installing on remote..."
"${SSH[@]}" "
set -euo pipefail
REMOTE_FLEX='${REMOTE_FLEX}'

cp /tmp/flexmopex_sync/__init__.py    \"\${REMOTE_FLEX}/__init__.py\"
cp /tmp/flexmopex_sync/run_model.py   \"\${REMOTE_FLEX}/run_model.py\"
cp /tmp/flexmopex_sync/pub_sampler.py \"\${REMOTE_FLEX}/models/pub_sampler.py\"
cp /tmp/flexmopex_sync/pub_trainer.py \"\${REMOTE_FLEX}/models/pub_trainer.py\"
cp /tmp/flexmopex_sync/run_block1_main.sh \"\${REMOTE_FLEX}/scripts/run_block1_main.sh\"
cp /tmp/flexmopex_sync/_worker_base_full.sh \"\${REMOTE_FLEX}/scripts/_worker_base_full.sh\"
cp /tmp/flexmopex_sync/_worker_flex_low.sh  \"\${REMOTE_FLEX}/scripts/_worker_flex_low.sh\"
cp /tmp/flexmopex_sync/_worker_flex_high.sh \"\${REMOTE_FLEX}/scripts/_worker_flex_high.sh\"
chmod +x \"\${REMOTE_FLEX}/scripts\"/_worker_*.sh \"\${REMOTE_FLEX}/scripts/run_block1_main.sh\"

cat > \"\${REMOTE_FLEX}/.env\" <<'ENV'
FLEXMOPEX_DATA_DIR=/root/autodl-fs
DATA_PATH=/root/autodl-fs
BASIN_GROUPS_DIR=/root/autodl-fs/basin_groups
GAGE_INFO=/root/autodl-fs/gage_id.npy
ENV

echo 'Syntax check shell workers...'
bash -n \"\${REMOTE_FLEX}/scripts/_worker_base_full.sh\"
bash -n \"\${REMOTE_FLEX}/scripts/_worker_flex_low.sh\"
bash -n \"\${REMOTE_FLEX}/scripts/_worker_flex_high.sh\"
echo 'Shell syntax OK'

mkdir -p \"\${REMOTE_FLEX}/results/block1_main\"
echo 'Install OK'
"

# ── DRY_RUN: one job, no tmux ─────────────────────────────────────────────────
if [[ "${DRY_RUN}" == "1" ]]; then
  echo ""
  echo "=== DRY_RUN: testing single job (base alpha=0.0 seed=42) — no tmux ==="
  "${SSH[@]}" "
set -euo pipefail
cd '${REMOTE_FLEX}'
export PATH='/root/miniconda3/bin':\$PATH
OUT_ROOT='results/block1_main'
SAVE='\${OUT_ROOT}/base/alpha0.0/seed42'
mkdir -p \"\${SAVE}\"
echo '[DRY_RUN] Running: python run_model.py --model-type base --alpha 0.0 --seed 42 --batch-size ${BATCH_SIZE}'
CUDA_VISIBLE_DEVICES=${GPU} python run_model.py \
  --model-type base \
  --alpha 0.0 \
  --seed 42 \
  --batch-size ${BATCH_SIZE} \
  --output-root \"\${SAVE}\" \
  2>&1 | tee \"\${SAVE}/train.log\"
echo '[DRY_RUN] Done.'
"
  exit 0
fi

# ── Launch tmux session with 3 windows ───────────────────────────────────────
echo "Launching tmux session: ${SESSION}"
"${SSH[@]}" "
set -euo pipefail
REMOTE_FLEX='${REMOTE_FLEX}'
SESSION='${SESSION}'

if ! command -v tmux >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q && apt-get install -y tmux
fi

if tmux has-session -t \"\${SESSION}\" 2>/dev/null; then
  echo 'tmux session already exists: '\"\${SESSION}\" >&2
  exit 1
fi

LOG_DIR=\"\${REMOTE_FLEX}/results/block1_main\"
mkdir -p \"\${LOG_DIR}\"

tmux new-session -d -s \"\${SESSION}\" -n 'base_full' \
  \"bash '\${REMOTE_FLEX}/scripts/_worker_base_full.sh' 2>&1 | tee '\${LOG_DIR}/tmux_base_full.log'; exec bash\"

tmux new-window -t \"\${SESSION}\" -n 'flex_low' \
  \"bash '\${REMOTE_FLEX}/scripts/_worker_flex_low.sh' 2>&1 | tee '\${LOG_DIR}/tmux_flex_low.log'; exec bash\"

tmux new-window -t \"\${SESSION}\" -n 'flex_high' \
  \"bash '\${REMOTE_FLEX}/scripts/_worker_flex_high.sh' 2>&1 | tee '\${LOG_DIR}/tmux_flex_high.log'; exec bash\"

tmux list-windows -t \"\${SESSION}\"
"

echo
echo "Started tmux session: ${SESSION}"
echo "  GPU=${GPU}, MAX_PARALLEL=${MAX_PARALLEL}, BATCH_SIZE=${BATCH_SIZE}"
echo
echo "Attach to full session:"
echo "  ssh -p ${PORT} ${REMOTE} \"tmux attach -t ${SESSION}\""
echo
echo "Attach to specific window:"
echo "  ssh -p ${PORT} ${REMOTE} \"tmux attach -t ${SESSION}:base_full\""
echo "  ssh -p ${PORT} ${REMOTE} \"tmux attach -t ${SESSION}:flex_low\""
echo "  ssh -p ${PORT} ${REMOTE} \"tmux attach -t ${SESSION}:flex_high\""
echo
echo "Monitor logs live:"
echo "  ssh -p ${PORT} ${REMOTE} \"tail -f ${REMOTE_FLEX}/results/block1_main/tmux_base_full.log\""
echo
echo "List windows:"
echo "  ssh -p ${PORT} ${REMOTE} \"tmux list-windows -t ${SESSION}\""
