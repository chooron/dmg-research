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
GPU="${GPU:-0}"
# 显存估算：单任务300MB → MAX_PARALLEL=24 约占 7.2GB，保守安全
MAX_PARALLEL="${MAX_PARALLEL:-10}"
# Key alphas × 5 seeds, path alphas × 3 seeds
KEY_ALPHAS="${KEY_ALPHAS:-0.005 0.01 0.03}"
KEY_SEEDS="${KEY_SEEDS:-42 123 456 789 1024}"
PATH_ALPHAS="${PATH_ALPHAS:-0.0 0.001 0.003 0.007 0.05 0.07 0.1}"
PATH_SEEDS="${PATH_SEEDS:-42 123 456}"
# DRY_RUN=1 → 跳过 tmux，直接在前台跑第一个 job（用于调试）
DRY_RUN="${DRY_RUN:-0}"

SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=no)
REMOTE="${USER}@${HOST}"

# ── SSH/SCP auth ──────────────────────────────────────────────────────────────
if [[ -z "${REMOTE_PASS:-}" && -f "$LOCAL_ENV" ]]; then
  REMOTE_PASS="$(sed -n 's/^SSH_PASSWORD=//p' "$LOCAL_ENV" | tail -n 1)"
fi

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
  cleanup() { rm -f "$ASKPASS_FILE" 2>/dev/null || true; }
  trap cleanup EXIT
  SSH=(env SSH_ASKPASS="$ASKPASS_FILE" SSH_ASKPASS_REQUIRE=force DISPLAY=dummy OMX_SSH_PASSWORD="$REMOTE_PASS" setsid -w ssh "${SSH_OPTS[@]}" "$REMOTE")
  SCP=(env SSH_ASKPASS="$ASKPASS_FILE" SSH_ASKPASS_REQUIRE=force DISPLAY=dummy OMX_SSH_PASSWORD="$REMOTE_PASS" setsid -w scp -P "$PORT" -o StrictHostKeyChecking=no)
else
  SSH=(ssh "${SSH_OPTS[@]}" "$REMOTE")
  SCP=(scp -P "$PORT" -o StrictHostKeyChecking=no)
fi

# ── Per-window parallelism ────────────────────────────────────────────────────
# Window A: key alphas (0.005, 0.01, 0.03) × 5 seeds = 15 jobs
PAR_A=$(( MAX_PARALLEL / 2 + 1 ))
# Window B: path alphas (7 values) × 3 seeds = 21 jobs
PAR_B=$(( MAX_PARALLEL / 2 ))
[[ $PAR_B -lt 1 ]] && PAR_B=1

# ── Generate worker scripts locally then SCP them ────────────────────────────
WORKER_DIR="$(mktemp -d)"
trap 'rm -rf "$WORKER_DIR"' EXIT

make_worker() {
  local name=$1 par=$2
  shift 2
  # remaining args: "alpha seed" pairs (space-separated, one per array element)
  local tasks=("$@")

  local out="$WORKER_DIR/${name}.sh"
  cat > "$out" <<HEADER
#!/usr/bin/env bash
set -euo pipefail
cd "\$(dirname "\$0")/.."
export PATH='/root/miniconda3/bin':\$PATH

GPU="${GPU}"
MAX_PARALLEL="${par}"
OUT_ROOT="results/block1_alpha_path"

run_exp() {
  local alpha=\$1 seed=\$2
  local tag="flex_alpha\${alpha}_seed\${seed}"
  local save_path="\${OUT_ROOT}/flex/alpha\${alpha}/seed\${seed}"
  mkdir -p "\${save_path}"
  echo "[\$(date +%H:%M:%S)] START  gpu=\${GPU}  \${tag}"
  CUDA_VISIBLE_DEVICES=\${GPU} python run_model.py \\
    --model-type flex \\
    --alpha "\${alpha}" \\
    --seed "\${seed}" \\
    --output-root "\${save_path}" \\
    > "\${save_path}/train.log" 2>&1
  echo "[\$(date +%H:%M:%S)] DONE   gpu=\${GPU}  \${tag}"
}

TASKS=(
HEADER

  for task_spec in "${tasks[@]}"; do
    echo "  \"${task_spec}\"" >> "$out"
  done

  cat >> "$out" <<'FOOTER'
)

PIDS=()
for task in "${TASKS[@]}"; do
  read -r alpha seed <<< "${task}"
  while [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]]; do
    NEW_PIDS=()
    for pid in "${PIDS[@]}"; do
      kill -0 "${pid}" 2>/dev/null && NEW_PIDS+=("${pid}")
    done
    PIDS=("${NEW_PIDS[@]+"${NEW_PIDS[@]}"}")
    [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]] && sleep 3
  done
  run_exp "${alpha}" "${seed}" &
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

# ── Build task lists ──────────────────────────────────────────────────────────
KEY_TASKS=()
for alpha in $KEY_ALPHAS; do
  for seed in $KEY_SEEDS; do
    KEY_TASKS+=("${alpha} ${seed}")
  done
done

PATH_TASKS=()
for alpha in $PATH_ALPHAS; do
  for seed in $PATH_SEEDS; do
    PATH_TASKS+=("${alpha} ${seed}")
  done
done

make_worker "_worker_key_alphas"  "$PAR_A" "${KEY_TASKS[@]}"
make_worker "_worker_path_alphas" "$PAR_B" "${PATH_TASKS[@]}"

# ── Upload all files ──────────────────────────────────────────────────────────
FILES=(
  project/flexmopex/__init__.py
  project/flexmopex/run_model.py
  project/flexmopex/models/pub_sampler.py
  project/flexmopex/models/pub_trainer.py
  project/flexmopex/scripts/run_block1_alpha_path.sh
)

echo "Creating remote sync dir..."
"${SSH[@]}" "rm -rf /tmp/flexmopex_sync && mkdir -p /tmp/flexmopex_sync"

echo "Uploading code..."
for f in "${FILES[@]}"; do
  "${SCP[@]}" "${LOCAL_ROOT}/${f}" "${REMOTE}:/tmp/flexmopex_sync/$(basename "$f")"
done

echo "Uploading worker scripts..."
for w in _worker_key_alphas _worker_path_alphas; do
  "${SCP[@]}" "${WORKER_DIR}/${w}.sh" "${REMOTE}:/tmp/flexmopex_sync/${w}.sh"
done

# ── Install on remote ─────────────────────────────────────────────────────────
SESSION="block1_alpha_$(date +%Y%m%d_%H%M%S)"

echo "Installing on remote..."
"${SSH[@]}" "
set -euo pipefail
REMOTE_FLEX='${REMOTE_FLEX}'

cp /tmp/flexmopex_sync/__init__.py    \"\${REMOTE_FLEX}/__init__.py\"
cp /tmp/flexmopex_sync/run_model.py   \"\${REMOTE_FLEX}/run_model.py\"
cp /tmp/flexmopex_sync/pub_sampler.py \"\${REMOTE_FLEX}/models/pub_sampler.py\"
cp /tmp/flexmopex_sync/pub_trainer.py \"\${REMOTE_FLEX}/models/pub_trainer.py\"
cp /tmp/flexmopex_sync/run_block1_alpha_path.sh \"\${REMOTE_FLEX}/scripts/run_block1_alpha_path.sh\"
cp /tmp/flexmopex_sync/_worker_key_alphas.sh  \"\${REMOTE_FLEX}/scripts/_worker_key_alphas.sh\"
cp /tmp/flexmopex_sync/_worker_path_alphas.sh \"\${REMOTE_FLEX}/scripts/_worker_path_alphas.sh\"
chmod +x \"\${REMOTE_FLEX}/scripts\"/_worker_*.sh \"\${REMOTE_FLEX}/scripts/run_block1_alpha_path.sh\"

cat > \"\${REMOTE_FLEX}/.env\" <<'ENV'
FLEXMOPEX_DATA_DIR=/root/autodl-fs
DATA_PATH=/root/autodl-fs
BASIN_GROUPS_DIR=/root/autodl-fs/basin_groups
GAGE_INFO=/root/autodl-fs/gage_id.npy
ENV

echo 'Syntax check shell workers...'
bash -n \"\${REMOTE_FLEX}/scripts/_worker_key_alphas.sh\"
bash -n \"\${REMOTE_FLEX}/scripts/_worker_path_alphas.sh\"
echo 'Shell syntax OK'

mkdir -p \"\${REMOTE_FLEX}/results/block1_alpha_path\"
echo 'Install OK'
"

# ── DRY_RUN: one job, no tmux ─────────────────────────────────────────────────
if [[ "${DRY_RUN}" == "1" ]]; then
  echo ""
  echo "=== DRY_RUN: testing single job (flex alpha=0.005 seed=42) — no tmux ==="
  "${SSH[@]}" "
set -euo pipefail
cd '${REMOTE_FLEX}'
export PATH='/root/miniconda3/bin':\$PATH
OUT_ROOT='results/block1_alpha_path'
SAVE='\${OUT_ROOT}/flex/alpha0.005/seed42'
mkdir -p \"\${SAVE}\"
echo '[DRY_RUN] Running: python run_model.py --model-type flex --alpha 0.005 --seed 42'
CUDA_VISIBLE_DEVICES=${GPU} python run_model.py \
  --model-type flex \
  --alpha 0.005 \
  --seed 42 \
  --output-root \"\${SAVE}\" \
  2>&1 | tee \"\${SAVE}/train.log\"
echo '[DRY_RUN] Done.'
"
  exit 0
fi

# ── Launch tmux session with 2 windows ───────────────────────────────────────
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

LOG_DIR=\"\${REMOTE_FLEX}/results/block1_alpha_path\"
mkdir -p \"\${LOG_DIR}\"

tmux new-session -d -s \"\${SESSION}\" -n 'key_alphas' \
  \"bash '\${REMOTE_FLEX}/scripts/_worker_key_alphas.sh' 2>&1 | tee '\${LOG_DIR}/tmux_key_alphas.log'; exec bash\"

tmux new-window -t \"\${SESSION}\" -n 'path_alphas' \
  \"bash '\${REMOTE_FLEX}/scripts/_worker_path_alphas.sh' 2>&1 | tee '\${LOG_DIR}/tmux_path_alphas.log'; exec bash\"

tmux list-windows -t \"\${SESSION}\"
"

echo
echo "Started tmux session: ${SESSION}"
echo
echo "Attach to full session:"
echo "  ssh -p ${PORT} ${REMOTE} \"tmux attach -t ${SESSION}\""
echo
echo "Attach to specific window:"
echo "  ssh -p ${PORT} ${REMOTE} \"tmux attach -t ${SESSION}:key_alphas\""
echo "  ssh -p ${PORT} ${REMOTE} \"tmux attach -t ${SESSION}:path_alphas\""
echo
echo "Monitor logs live:"
echo "  ssh -p ${PORT} ${REMOTE} \"tail -f ${REMOTE_FLEX}/results/block1_alpha_path/tmux_key_alphas.log\""
echo "  ssh -p ${PORT} ${REMOTE} \"tail -f ${REMOTE_FLEX}/results/block1_alpha_path/tmux_path_alphas.log\""
echo
echo "List windows:"
echo "  ssh -p ${PORT} ${REMOTE} \"tmux list-windows -t ${SESSION}\""
