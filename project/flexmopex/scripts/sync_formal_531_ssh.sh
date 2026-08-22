#!/usr/bin/env bash
# Sync formal 531 code/configs/manifests to the existing SSH workspace.
# Results/logs/outputs are deliberately excluded and are never deleted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HOST="${HOST:-connect.nmb2.seetacloud.com}"
PORT="${PORT:-33933}"
USER_NAME="${USER_NAME:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/dmg-research}"
REMOTE_FLEX="${REMOTE_ROOT}/project/flexmopex"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-/root/autodl-fs/data}"
REMOTE_PASS="${REMOTE_PASS:-}"

if [[ -z "${REMOTE_PASS}" ]]; then
  echo "Set REMOTE_PASS in the calling environment; it is intentionally not stored in this script." >&2
  exit 2
fi

ASKPASS_FILE="$(mktemp)"
cleanup() { rm -f "${ASKPASS_FILE}"; }
trap cleanup EXIT
printf '#!/usr/bin/env bash\nprintf %%s %q\n' "${REMOTE_PASS}" >"${ASKPASS_FILE}"
chmod 700 "${ASKPASS_FILE}"

export SSH_ASKPASS="${ASKPASS_FILE}"
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=dummy
SSH_BASE=(ssh -F /dev/null -p "${PORT}" -o StrictHostKeyChecking=no)
RSYNC_RSH="setsid -w ssh -F /dev/null -p ${PORT} -o StrictHostKeyChecking=no"
REMOTE="${USER_NAME}@${HOST}"

"${SSH_BASE[@]}" "${REMOTE}" "mkdir -p '${REMOTE_FLEX}/models' '${REMOTE_FLEX}/scripts' '${REMOTE_FLEX}/test' '${REMOTE_FLEX}/manifests'"

rsync -az -e "${RSYNC_RSH}" \
  --exclude='results/' --exclude='logs/' --exclude='outputs/' \
  --exclude='.tmp/' --exclude='__pycache__/' --exclude='*.pyc' \
  "${LOCAL_ROOT}/project/flexmopex/" "${REMOTE}:${REMOTE_FLEX}/"
rsync -az -e "${RSYNC_RSH}" \
  "${LOCAL_ROOT}/project/bettermodel/implements/my_trainer.py" \
  "${REMOTE}:${REMOTE_ROOT}/project/bettermodel/implements/my_trainer.py"
rsync -az -e "${RSYNC_RSH}" \
  "${LOCAL_ROOT}/data/531sub_id.txt" \
  "${REMOTE}:${REMOTE_ROOT}/data/531sub_id.txt"

"${SSH_BASE[@]}" "${REMOTE}" bash -s <<REMOTE_SETUP
set -euo pipefail
REMOTE_FLEX='${REMOTE_FLEX}'
REMOTE_DATA_DIR='${REMOTE_DATA_DIR}'
cat > "\${REMOTE_FLEX}/.env" <<ENV
FLEXMOPEX_DATA_DIR=\${REMOTE_DATA_DIR}
DATA_PATH=\${REMOTE_DATA_DIR}
BASIN_GROUPS_DIR=\${REMOTE_DATA_DIR}/basin_groups
GAGE_INFO=\${REMOTE_DATA_DIR}/gage_id.npy
ENV
export PATH='/root/miniconda3/bin':\$PATH
cd "\${REMOTE_FLEX}"
/root/miniconda3/bin/python -m py_compile \
  run_model.py models/early_stopping.py models/cf_trainer.py \
  models/pub_trainer.py models/pub_sampler.py \
  models/learned_weight_mopex_candidates.py \
  scripts/build_531_loro_groups.py
bash -n scripts/run_formal_531_staged.sh
bash -n scripts/sync_formal_531_ssh.sh
printf 'REMOTE_SYNC_VALIDATION_OK\\n'
REMOTE_SETUP

printf 'SYNC_CHECKSUMS\\n'
for file in \
  project/flexmopex/run_model.py \
  project/flexmopex/models/early_stopping.py \
  project/flexmopex/models/cf_trainer.py \
  project/flexmopex/models/pub_trainer.py \
  project/flexmopex/models/pub_sampler.py \
  project/flexmopex/scripts/run_formal_531_staged.sh \
  project/flexmopex/formal_experiment_manifest.yaml; do
  local_hash="$(sha256sum "${LOCAL_ROOT}/${file}" | awk '{print $1}')"
  remote_hash="$("${SSH_BASE[@]}" "${REMOTE}" "sha256sum '${REMOTE_ROOT}/${file}' | awk '{print \$1}'")"
  printf '%s local=%s remote=%s\n' "${file}" "${local_hash}" "${remote_hash}"
  [[ "${local_hash}" == "${remote_hash}" ]]
done
printf 'FORMAL_531_SSH_SYNC_OK host=%s port=%s root=%s\n' "${HOST}" "${PORT}" "${REMOTE_ROOT}"
