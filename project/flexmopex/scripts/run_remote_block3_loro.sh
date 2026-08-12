#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-connect.westb.seetacloud.com}"
PORT="${PORT:-15632}"
USER="${USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/dmg-research}"
REMOTE_FLEX="${REMOTE_ROOT}/project/flexmopex"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/miniconda3/bin/python}"
LOCAL_ROOT="${LOCAL_ROOT:-/workspace/autoresearch}"
LOCAL_ENV="${LOCAL_ENV:-${LOCAL_ROOT}/project/flexmopex/.env}"

GPU="${GPU:-0}"
PER_REGION_PARALLEL="${PER_REGION_PARALLEL:-1}"
LORO_SEEDS="${LORO_SEEDS:-42}"
LORO_MODEL_TYPES="${LORO_MODEL_TYPES:-flex full base}"
LORO_REGIONS="${LORO_REGIONS:-0 1 2 3 4 5 6}"
SESSION="${SESSION:-block3_loro}"

SSH_OPTS=(-F /dev/null -p "$PORT" -o StrictHostKeyChecking=no)
SCP_OPTS=(-F /dev/null -P "$PORT" -o StrictHostKeyChecking=no)
REMOTE="${USER}@${HOST}"

if [[ -z "${REMOTE_PASS:-}" && -f "$LOCAL_ENV" ]]; then
  REMOTE_PASS="$(sed -n 's/^SSH_PASSWORD=//p' "$LOCAL_ENV" | tail -n 1)"
fi

FILES=(
  project/flexmopex/__init__.py
  project/flexmopex/run_model.py
  project/flexmopex/run_batch_loro.py
  project/flexmopex/models/pub_sampler.py
  project/flexmopex/models/pub_trainer.py
  project/flexmopex/scripts/run_block3_loro.sh
  project/flexmopex/scripts/run_block3_loro_batch.sh
)

if command -v sshpass >/dev/null 2>&1 && [[ -n "${REMOTE_PASS:-}" ]]; then
  SSH=(sshpass -p "$REMOTE_PASS" ssh "${SSH_OPTS[@]}" "$REMOTE")
  SCP=(sshpass -p "$REMOTE_PASS" scp "${SCP_OPTS[@]}")
elif [[ -n "${REMOTE_PASS:-}" ]]; then
  ASKPASS_FILE="$(mktemp)"
  cat > "$ASKPASS_FILE" <<'ASKPASS'
#!/usr/bin/env bash
printf "%s" "$OMX_SSH_PASSWORD"
ASKPASS
  chmod 700 "$ASKPASS_FILE"
  cleanup() {
    rm -f "$ASKPASS_FILE"
  }
  trap cleanup EXIT
  SSH=(env SSH_ASKPASS="$ASKPASS_FILE" SSH_ASKPASS_REQUIRE=force DISPLAY=dummy OMX_SSH_PASSWORD="$REMOTE_PASS" setsid -w ssh "${SSH_OPTS[@]}" "$REMOTE")
  SCP=(env SSH_ASKPASS="$ASKPASS_FILE" SSH_ASKPASS_REQUIRE=force DISPLAY=dummy OMX_SSH_PASSWORD="$REMOTE_PASS" setsid -w scp "${SCP_OPTS[@]}")
else
  SSH=(ssh "${SSH_OPTS[@]}" "$REMOTE")
  SCP=(scp "${SCP_OPTS[@]}")
fi

echo "Creating remote sync dir..."
"${SSH[@]}" "rm -rf /tmp/flexmopex_sync && mkdir -p /tmp/flexmopex_sync"

echo "Uploading code..."
for f in "${FILES[@]}"; do
  "${SCP[@]}" "${LOCAL_ROOT}/${f}" "${REMOTE}:/tmp/flexmopex_sync/$(basename "$f")"
done

echo "Installing code and starting tmux session: ${SESSION}"
"${SSH[@]}" bash -s <<REMOTE
set -euo pipefail

REMOTE_FLEX="${REMOTE_FLEX}"
REMOTE_PYTHON="${REMOTE_PYTHON}"
GPU="${GPU}"
PER_REGION_PARALLEL="${PER_REGION_PARALLEL}"
LORO_SEEDS="${LORO_SEEDS}"
LORO_MODEL_TYPES="${LORO_MODEL_TYPES}"
LORO_REGIONS="${LORO_REGIONS}"
SESSION="${SESSION}"

cp /tmp/flexmopex_sync/__init__.py "\${REMOTE_FLEX}/__init__.py"
cp /tmp/flexmopex_sync/run_model.py "\${REMOTE_FLEX}/run_model.py"
cp /tmp/flexmopex_sync/run_batch_loro.py "\${REMOTE_FLEX}/run_batch_loro.py"
cp /tmp/flexmopex_sync/pub_sampler.py "\${REMOTE_FLEX}/models/pub_sampler.py"
cp /tmp/flexmopex_sync/pub_trainer.py "\${REMOTE_FLEX}/models/pub_trainer.py"
cp /tmp/flexmopex_sync/run_block3_loro.sh "\${REMOTE_FLEX}/scripts/run_block3_loro.sh"
cp /tmp/flexmopex_sync/run_block3_loro_batch.sh "\${REMOTE_FLEX}/scripts/run_block3_loro_batch.sh"
chmod +x "\${REMOTE_FLEX}/scripts/run_block3_loro.sh"
chmod +x "\${REMOTE_FLEX}/scripts/run_block3_loro_batch.sh"

cat > "\${REMOTE_FLEX}/.env" <<'ENV'
FLEXMOPEX_DATA_DIR=/root/autodl-fs
DATA_PATH=/root/autodl-fs
BASIN_GROUPS_DIR=/root/autodl-fs/basin_groups
GAGE_INFO=/root/autodl-fs/gage_id.npy
ENV

if [[ ! -x "\${REMOTE_PYTHON}" ]]; then
  REMOTE_PYTHON="\$(command -v python3 || command -v python)"
fi

"\${REMOTE_PYTHON}" -m py_compile \
  "\${REMOTE_FLEX}/__init__.py" \
  "\${REMOTE_FLEX}/run_model.py" \
  "\${REMOTE_FLEX}/run_batch_loro.py" \
  "\${REMOTE_FLEX}/models/pub_sampler.py" \
  "\${REMOTE_FLEX}/models/pub_trainer.py"

bash -n "\${REMOTE_FLEX}/scripts/run_block3_loro_batch.sh"
# Persistent compile cache dir
export TORCHINDUCTOR_CACHE_DIR="\${TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_inductor_cache}"
mkdir -p "\${TORCHINDUCTOR_CACHE_DIR}"

mkdir -p "\${REMOTE_FLEX}/results/block3_loro"
# Pre-create hierarchical directories: {model_type}/region{r}/seed{seed}
for mt in \${LORO_MODEL_TYPES}; do
  for r in 0 1 2 3 4 5 6; do
    for s in \${LORO_SEEDS}; do
      mkdir -p "\${REMOTE_FLEX}/results/block3_loro/\${mt}/region\${r}/seed\${s}"
    done
  done
done

if ! command -v tmux >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y tmux
fi

if tmux has-session -t "\${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: \${SESSION}" >&2
  exit 1
fi

# Batch mode: single tmux window runs all regions sequentially (or with limited
# parallelism via MAX_PARALLEL). No GPU memory contention from 7 concurrent windows.
tmux new-session -d -s "\${SESSION}" -n "batch" \
  "cd '\${REMOTE_FLEX}' && \
   export PATH='/root/miniconda3/bin:'\\\$PATH && \
   export TORCHINDUCTOR_CACHE_DIR='\${TORCHINDUCTOR_CACHE_DIR}' && \
   PYTHON_BIN='\${REMOTE_PYTHON}' \
   LORO_SEEDS='\${LORO_SEEDS}' \
   LORO_MODEL_TYPES='\${LORO_MODEL_TYPES}' \
   LORO_REGIONS='\${LORO_REGIONS}' \
   bash scripts/run_block3_loro_batch.sh '\${GPU}' '\${PER_REGION_PARALLEL}' \
   2>&1 | tee results/block3_loro/tmux_batch.log"

tmux list-windows -t "\${SESSION}"
REMOTE

echo
echo "Started tmux session: ${SESSION}"
echo "Attach:"
echo "ssh -p ${PORT} ${REMOTE} \"tmux attach -t ${SESSION}\""
echo
echo "Check windows:"
echo "ssh -p ${PORT} ${REMOTE} \"tmux list-windows -t ${SESSION}\""
