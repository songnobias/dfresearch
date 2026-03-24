#!/usr/bin/env bash
#
# End-to-end image model submission for BitMind SN34.
#
# Runs inside the dfresearch directory. Requires:
#   - dfresearch venv with deps installed (uv sync)
#   - NVIDIA GPU for training
#   - bitmind-subnet installed (for gasbench + gascli)
#   - Registered wallet/hotkey on SN34
#
# Usage:
#   ./scripts/submit_image.sh                           # full pipeline, default model
#   ./scripts/submit_image.sh --model convnext-base     # pick architecture
#   ./scripts/submit_image.sh --skip-train              # export+validate+push only
#   ./scripts/submit_image.sh --skip-validate           # skip gasbench local check
#   ./scripts/submit_image.sh --dry-run                 # everything except push
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBNET_ROOT="$(cd "${DF_ROOT}/.." && pwd)"

# ── Defaults ──
MODEL="efficientnet-b4"
WALLET_NAME="default"
WALLET_HOTKEY="default"
SKIP_TRAIN=false
SKIP_VALIDATE=false
DRY_RUN=false
HOURS=""

# ── Parse args ──
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="$2";        shift 2 ;;
    --wallet-name)  WALLET_NAME="$2";  shift 2 ;;
    --wallet-hotkey) WALLET_HOTKEY="$2"; shift 2 ;;
    --skip-train)   SKIP_TRAIN=true;   shift ;;
    --skip-validate) SKIP_VALIDATE=true; shift ;;
    --dry-run)      DRY_RUN=true;      shift ;;
    --hours)        HOURS="$2";        shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

CKPT_DIR="${DF_ROOT}/results/checkpoints/image"
EXPORT_DIR="${DF_ROOT}/results/exports"
ZIP_NAME="image_detector_${MODEL}.zip"
ZIP_PATH="${EXPORT_DIR}/${ZIP_NAME}"

echo "============================================================"
echo "  BitMind SN34 — Image Model Submission"
echo "============================================================"
echo "  Model:        ${MODEL}"
echo "  Wallet:       ${WALLET_NAME} / ${WALLET_HOTKEY}"
echo "  dfresearch:   ${DF_ROOT}"
echo "  bitmind-subnet: ${SUBNET_ROOT}"
echo "============================================================"
echo ""

# ── Step 1: Prepare data (if cache is empty) ──
echo "[1/5] Checking training data..."
cd "${DF_ROOT}"
source .venv/bin/activate 2>/dev/null || { echo "No dfresearch .venv — run: cd ${DF_ROOT} && uv venv --python 3.11 && uv sync"; exit 1; }

CACHE_DIR="${HOME}/.cache/dfresearch/datasets/image"
if [[ -d "${CACHE_DIR}" ]] && [[ $(find "${CACHE_DIR}" -name "*.png" 2>/dev/null | head -1) ]]; then
  echo "  Data cache found at ${CACHE_DIR}"
else
  echo "  Downloading image datasets (500 samples/dataset)..."
  uv run prepare.py --modality image
fi

# ── Step 2: Train ──
if [[ "${SKIP_TRAIN}" == "false" ]]; then
  echo ""
  echo "[2/5] Training ${MODEL}..."
  if [[ -n "${HOURS}" ]]; then
    echo "  Full training for ${HOURS} hours..."
    uv run train_full.py --modality image --model "${MODEL}" --hours "${HOURS}"
  else
    echo "  Quick training (10 min budget)..."
    uv run train_image.py --model "${MODEL}"
  fi
else
  echo ""
  echo "[2/5] Skipping training (--skip-train)"
  if [[ ! -f "${CKPT_DIR}/model.safetensors" ]]; then
    echo "  ERROR: No checkpoint at ${CKPT_DIR}/model.safetensors"
    echo "  Run training first or remove --skip-train"
    exit 1
  fi
fi

# ── Step 3: Export ZIP ──
echo ""
echo "[3/5] Exporting submission ZIP..."
uv run export.py --modality image --model "${MODEL}"
echo "  ZIP: ${ZIP_PATH}"

# ── Step 4: Validate with gasbench ──
if [[ "${SKIP_VALIDATE}" == "false" ]]; then
  echo ""
  echo "[4/5] Validating with gasbench (--small, mirrors entrance exam)..."

  # gasbench expects a directory, not a zip — unzip to a temp dir
  VALIDATE_DIR=$(mktemp -d)
  unzip -q "${ZIP_PATH}" -d "${VALIDATE_DIR}"

  # Use bitmind-subnet's venv for gasbench
  if [[ -f "${SUBNET_ROOT}/.venv/bin/gasbench" ]]; then
    "${SUBNET_ROOT}/.venv/bin/gasbench" run --image-model "${VALIDATE_DIR}" --small
  else
    echo "  WARNING: gasbench not found at ${SUBNET_ROOT}/.venv/bin/gasbench"
    echo "  Install bitmind-subnet and gasbench to validate locally."
    echo "  Skipping validation."
  fi
  rm -rf "${VALIDATE_DIR}"
else
  echo ""
  echo "[4/5] Skipping validation (--skip-validate)"
fi

# ── Step 5: Push to SN34 ──
echo ""
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[5/5] DRY RUN — not pushing. Command would be:"
  echo "  gascli d push \\"
  echo "    --image-model ${ZIP_PATH} \\"
  echo "    --wallet-name ${WALLET_NAME} \\"
  echo "    --wallet-hotkey ${WALLET_HOTKEY}"
else
  echo "[5/5] Pushing to BitMind Subnet 34..."
  if [[ -f "${SUBNET_ROOT}/.venv/bin/gascli" ]]; then
    "${SUBNET_ROOT}/.venv/bin/gascli" d push \
      --image-model "${ZIP_PATH}" \
      --wallet-name "${WALLET_NAME}" \
      --wallet-hotkey "${WALLET_HOTKEY}"
  else
    echo "  ERROR: gascli not found at ${SUBNET_ROOT}/.venv/bin/gascli"
    echo "  Install bitmind-subnet first: cd ${SUBNET_ROOT} && ./install.sh --no-system-deps"
    exit 1
  fi
fi

echo ""
echo "============================================================"
echo "  Done."
echo "  ZIP: ${ZIP_PATH}"
echo "  After push: watch status at https://app.bitmind.ai/"
echo "============================================================"
