#!/usr/bin/env bash
#
# Run this ON YOUR CPU VPS after training is done on the GPU machine.
# It pulls the ZIP from the GPU, validates locally with gasbench, and pushes to SN34.
#
# Usage:
#   # Pull from GPU machine + validate + push:
#   bash scripts/vps_validate_and_push.sh \
#     --gpu-host ubuntu@gpu-server \
#     --gpu-path /home/ubuntu/dfresearch \
#     --model convnext-base \
#     --wallet-name mywallet \
#     --wallet-hotkey myhotkey
#
#   # If you already copied the ZIP manually:
#   bash scripts/vps_validate_and_push.sh \
#     --zip /path/to/image_detector_convnext-base.zip \
#     --wallet-name mywallet \
#     --wallet-hotkey myhotkey
#
#   # Dry run (validate only, don't push):
#   bash scripts/vps_validate_and_push.sh \
#     --zip /path/to/zip \
#     --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBNET_ROOT="$(cd "${DF_ROOT}/.." && pwd)"

# ── Defaults ──
GPU_HOST=""
GPU_PATH=""
MODEL="convnext-base"
ZIP_PATH=""
WALLET_NAME="default"
WALLET_HOTKEY="default"
DRY_RUN=false
SKIP_VALIDATE=false
SSH_KEY=""

# ── Parse args ──
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-host)      GPU_HOST="$2";      shift 2 ;;
    --gpu-path)      GPU_PATH="$2";      shift 2 ;;
    --model)         MODEL="$2";         shift 2 ;;
    --zip)           ZIP_PATH="$2";      shift 2 ;;
    --wallet-name)   WALLET_NAME="$2";   shift 2 ;;
    --wallet-hotkey) WALLET_HOTKEY="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=true;       shift ;;
    --skip-validate) SKIP_VALIDATE=true; shift ;;
    --ssh-key)       SSH_KEY="$2";       shift 2 ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --gpu-host USER@HOST   GPU machine SSH address"
      echo "  --gpu-path PATH        dfresearch path on GPU machine"
      echo "  --model NAME           Model name (default: convnext-base)"
      echo "  --zip PATH             Use existing local ZIP instead of pulling from GPU"
      echo "  --wallet-name NAME     Bittensor wallet name"
      echo "  --wallet-hotkey NAME   Bittensor hotkey name"
      echo "  --dry-run              Validate only, don't push"
      echo "  --skip-validate        Skip gasbench validation"
      echo "  --ssh-key PATH         SSH key file for GPU connection"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

ZIP_NAME="image_detector_${MODEL}.zip"
LOCAL_ZIP="${DF_ROOT}/results/exports/${ZIP_NAME}"

echo "============================================================"
echo "  VPS — Validate & Push to BitMind SN34"
echo "============================================================"

# ── Step 1: Get the ZIP ──
if [[ -n "${ZIP_PATH}" ]]; then
  echo "[1/3] Using provided ZIP: ${ZIP_PATH}"
  LOCAL_ZIP="${ZIP_PATH}"
elif [[ -n "${GPU_HOST}" && -n "${GPU_PATH}" ]]; then
  echo "[1/3] Pulling ZIP from ${GPU_HOST}..."
  mkdir -p "${DF_ROOT}/results/exports"

  SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
  if [[ -n "${SSH_KEY}" ]]; then
    SSH_OPTS+=(-i "${SSH_KEY}")
  fi

  REMOTE_ZIP="${GPU_PATH}/results/exports/${ZIP_NAME}"
  scp "${SSH_OPTS[@]}" "${GPU_HOST}:${REMOTE_ZIP}" "${LOCAL_ZIP}"
  echo "  Downloaded to: ${LOCAL_ZIP}"
elif [[ -f "${LOCAL_ZIP}" ]]; then
  echo "[1/3] Found existing ZIP: ${LOCAL_ZIP}"
else
  echo "[1/3] ERROR: No ZIP found. Provide --gpu-host + --gpu-path, or --zip, or train first."
  exit 1
fi

# Verify ZIP exists and has content
if [[ ! -f "${LOCAL_ZIP}" ]]; then
  echo "  ERROR: ZIP not found at ${LOCAL_ZIP}"
  exit 1
fi
echo "  ZIP size: $(du -h "${LOCAL_ZIP}" | cut -f1)"
echo "  Contents:"
unzip -l "${LOCAL_ZIP}" | tail -n +4 | head -n -2 | while read -r line; do echo "    ${line}"; done

# ── Step 2: Validate with gasbench ──
if [[ "${SKIP_VALIDATE}" == "false" ]]; then
  echo ""
  echo "[2/3] Validating with gasbench (--small = entrance exam mirror)..."

  GASBENCH_BIN="${SUBNET_ROOT}/.venv/bin/gasbench"
  if [[ ! -f "${GASBENCH_BIN}" ]]; then
    echo "  WARNING: gasbench not found at ${GASBENCH_BIN}"
    echo "  Install: cd ${SUBNET_ROOT} && ./install.sh --no-system-deps"
    echo "  Skipping validation."
  else
    VALIDATE_DIR=$(mktemp -d)
    unzip -q "${LOCAL_ZIP}" -d "${VALIDATE_DIR}"
    echo "  Running gasbench..."
    "${GASBENCH_BIN}" run --image-model "${VALIDATE_DIR}" --small || {
      echo "  WARNING: gasbench returned non-zero. Check output above."
    }
    rm -rf "${VALIDATE_DIR}"
  fi
else
  echo ""
  echo "[2/3] Skipping validation (--skip-validate)"
fi

# ── Step 3: Push to SN34 ──
echo ""
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[3/3] DRY RUN — not pushing. Command would be:"
  echo ""
  echo "  gascli d push \\"
  echo "    --image-model ${LOCAL_ZIP} \\"
  echo "    --wallet-name ${WALLET_NAME} \\"
  echo "    --wallet-hotkey ${WALLET_HOTKEY}"
else
  echo "[3/3] Pushing to BitMind Subnet 34..."

  GASCLI_BIN="${SUBNET_ROOT}/.venv/bin/gascli"
  if [[ ! -f "${GASCLI_BIN}" ]]; then
    echo "  ERROR: gascli not found at ${GASCLI_BIN}"
    echo "  Install: cd ${SUBNET_ROOT} && ./install.sh --no-system-deps"
    exit 1
  fi

  "${GASCLI_BIN}" d push \
    --image-model "${LOCAL_ZIP}" \
    --wallet-name "${WALLET_NAME}" \
    --wallet-hotkey "${WALLET_HOTKEY}"
fi

echo ""
echo "============================================================"
echo "  Done."
echo "  After push: watch status at https://app.bitmind.ai/"
echo "============================================================"
