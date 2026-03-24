#!/usr/bin/env bash
# Run dfresearch training on a remote GPU machine from your CPU VPS (or laptop).
#
# Flow:
#   1) rsync project sources to REMOTE (excludes .venv, caches, large artifacts)
#   2) ssh: uv sync + optional prepare + train_image.py (or train_full.py)
#   3) rsync results/checkpoints (and optional runs/) back to LOCAL
#
# Prereqs on BOTH sides: rsync, ssh, same major Python if you rely on local venv (we rebuild on GPU).
# On REMOTE: NVIDIA driver, CUDA, uv, git; copy .env with HF token to REMOTE_DFRESEARCH_ROOT/.env
#
# Usage:
#   chmod +x scripts/remote_gpu_train.sh
#   cp scripts/remote_gpu_config.env.example scripts/remote_gpu_config.env
#   # edit remote_gpu_config.env
#   ./scripts/remote_gpu_train.sh train_image --model efficientnet-b4
#   ./scripts/remote_gpu_train.sh prepare --modality image
#   ./scripts/remote_gpu_train.sh export --modality image --model efficientnet-b4
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${DF_ROOT}/scripts/remote_gpu_config.env"

if [[ -f "${CONFIG}" ]]; then
  # shellcheck source=/dev/null
  set -a && source "${CONFIG}" && set +a
fi

: "${REMOTE_USER_HOST:?Set REMOTE_USER_HOST in scripts/remote_gpu_config.env}"
: "${REMOTE_DFRESEARCH_ROOT:?Set REMOTE_DFRESEARCH_ROOT in scripts/remote_gpu_config.env}"

LOCAL_ROOT="${LOCAL_DFRESEARCH_ROOT:-${DF_ROOT}}"

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [[ -n "${SSH_IDENTITY_FILE:-}" ]]; then
  SSH_OPTS+=(-i "${SSH_IDENTITY_FILE}")
fi

RSYNC_OPTS=(-avz --delete --partial)
if [[ -n "${RSYNC_BANDWIDTH_LIMIT_KB:-}" ]]; then
  RSYNC_OPTS+=("--bwlimit=${RSYNC_BANDWIDTH_LIMIT_KB}")
fi
RSYNC_OPTS+=("${RSYNC_EXTRA:-}")

# Keep GPU cache of HF + dfresearch datasets on the remote; do not wipe them from local sync.
RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.env'
  --exclude 'runs/'
  --exclude 'results/'
  --exclude '.ruff_cache/'
  --exclude '*.log'
)

remote_sh() {
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER_HOST}" "$@"
}

sync_to_remote() {
  echo "[sync] Local -> ${REMOTE_USER_HOST}:${REMOTE_DFRESEARCH_ROOT}"
  rsync "${RSYNC_OPTS[@]}" "${RSYNC_EXCLUDES[@]}" \
    "${LOCAL_ROOT}/" \
    "${REMOTE_USER_HOST}:${REMOTE_DFRESEARCH_ROOT}/"
}

sync_from_remote() {
  echo "[sync] ${REMOTE_USER_HOST}:${REMOTE_DFRESEARCH_ROOT}/results -> ${LOCAL_ROOT}/results"
  mkdir -p "${LOCAL_ROOT}/results"
  rsync "${RSYNC_OPTS[@]}" \
    "${REMOTE_USER_HOST}:${REMOTE_DFRESEARCH_ROOT}/results/" \
    "${LOCAL_ROOT}/results/"

  echo "[sync] ${REMOTE_USER_HOST}:${REMOTE_DFRESEARCH_ROOT}/runs (optional) -> ${LOCAL_ROOT}/runs"
  mkdir -p "${LOCAL_ROOT}/runs"
  rsync "${RSYNC_OPTS[@]}" \
    "${REMOTE_USER_HOST}:${REMOTE_DFRESEARCH_ROOT}/runs/" \
    "${LOCAL_ROOT}/runs/" || true
}

run_remote_command() {
  # Args: remote bash command string
  remote_sh bash -lc "$1"
}

# Map wrapper name -> dfresearch entry script (uv run …)
entry_script() {
  case "$1" in
    train_image) echo "train_image.py" ;;
    train_video) echo "train_video.py" ;;
    train_audio) echo "train_audio.py" ;;
    train_full) echo "train_full.py" ;;
    prepare) echo "prepare.py" ;;
    evaluate) echo "evaluate.py" ;;
    export) echo "export.py" ;;
    *) echo "" ;;
  esac
}

REMOTE_ROOT_Q=$(printf '%q' "${REMOTE_DFRESEARCH_ROOT}")

cmd="${1:-}"
shift || true

case "${cmd}" in
  sync-up)
    sync_to_remote
    ;;
  sync-down)
    sync_from_remote
    ;;
  train_image|train_video|train_audio|train_full|prepare|evaluate|export)
    py=$(entry_script "${cmd}")
    if [[ -z "${py}" ]]; then
      echo "Unknown command: ${cmd}" >&2
      exit 1
    fi
    sync_to_remote
    ARGS_Q=$(printf '%q ' "$@")
    # shellcheck disable=SC2086
    run_remote_command "set -euo pipefail
cd ${REMOTE_ROOT_Q}
if [[ ! -d .venv ]]; then uv venv --python 3.11; fi
source .venv/bin/activate
uv sync
uv run ${py} ${ARGS_Q}"
    sync_from_remote
    echo "[done] Check ${LOCAL_ROOT}/results/checkpoints/ and results/exports/"
    ;;
  shell)
    sync_to_remote
    remote_sh -t bash -lc "cd ${REMOTE_ROOT_Q} && exec bash -l"
    ;;
  *)
    echo "Usage: $0 {sync-up|sync-down|train_image|train_video|train_audio|train_full|prepare|evaluate|export|shell} [args...]"
    echo ""
    echo "Examples:"
    echo "  $0 prepare --modality image"
    echo "  $0 train_image --model efficientnet-b4"
    echo "  $0 export --modality image --model efficientnet-b4"
    echo "  $0 train_full --modality image --hours 4 --max-samples 2000"
    exit 1
    ;;
esac
