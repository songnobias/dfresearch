#!/usr/bin/env bash
#
# Run this DIRECTLY ON YOUR GPU MACHINE (via SSH).
# It handles everything: install, data download, train, export.
#
# Usage:
#   # First time — full setup + train:
#   bash gpu_setup_and_train.sh --hf-token hf_XXXX
#
#   # Subsequent runs — just train a different model:
#   bash gpu_setup_and_train.sh --skip-setup --model clip-vit-l14
#
#   # Long training run after you found good hyperparams:
#   bash gpu_setup_and_train.sh --skip-setup --model convnext-base --hours 4
#
set -euo pipefail

# ── Defaults ──
MODEL="convnext-base"
HF_TOKEN=""
SKIP_SETUP=false
HOURS=""
MAX_SAMPLES=500
DFRESEARCH_DIR="${HOME}/dfresearch"
REPO_URL="https://github.com/BitMind-AI/dfresearch.git"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="$2";          shift 2 ;;
    --hf-token)     HF_TOKEN="$2";       shift 2 ;;
    --skip-setup)   SKIP_SETUP=true;     shift ;;
    --hours)        HOURS="$2";          shift 2 ;;
    --max-samples)  MAX_SAMPLES="$2";    shift 2 ;;
    --dir)          DFRESEARCH_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --model NAME       Model to train (default: convnext-base)"
      echo "                     Options: efficientnet-b4, convnext-base, clip-vit-l14, smogy-swin"
      echo "  --hf-token TOKEN   HuggingFace token (required first time)"
      echo "  --skip-setup       Skip install + data download"
      echo "  --hours N          Long training (N hours). Default: 10 min quick train"
      echo "  --max-samples N    Samples per dataset (default: 500, use 2000+ for long runs)"
      echo "  --dir PATH         dfresearch directory (default: ~/dfresearch)"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "============================================================"
echo "  GPU Training — BitMind SN34 Image Detector"
echo "============================================================"
echo "  Model:      ${MODEL}"
echo "  Directory:  ${DFRESEARCH_DIR}"
echo "  Mode:       $(if [[ -n "${HOURS}" ]]; then echo "Full (${HOURS}h)"; else echo "Quick (10 min)"; fi)"
echo "============================================================"
echo ""

# ── Step 1: Check GPU ──
echo "[1/5] Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "  ERROR: nvidia-smi not found. Install NVIDIA drivers + CUDA first."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Step 2: Install (first time only) ──
if [[ "${SKIP_SETUP}" == "false" ]]; then
  echo "[2/5] Setting up dfresearch..."

  # Install uv if missing
  if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  # Clone or update repo
  if [[ -d "${DFRESEARCH_DIR}/.git" ]]; then
    echo "  Updating existing repo..."
    cd "${DFRESEARCH_DIR}"
    git pull --ff-only || true
  elif [[ -d "${DFRESEARCH_DIR}" ]]; then
    echo "  Directory exists (not a git repo), using as-is..."
    cd "${DFRESEARCH_DIR}"
  else
    echo "  Cloning dfresearch..."
    git clone "${REPO_URL}" "${DFRESEARCH_DIR}"
    cd "${DFRESEARCH_DIR}"
  fi

  # Create venv + install deps
  if [[ ! -d .venv ]]; then
    echo "  Creating virtual environment..."
    uv venv --python 3.11
  fi
  source .venv/bin/activate
  echo "  Installing dependencies..."
  uv sync

  # Set up .env with HF token
  if [[ -n "${HF_TOKEN}" ]]; then
    echo "HF_TOKEN=${HF_TOKEN}" > .env
    echo "  HF token saved to .env"
  elif [[ ! -f .env ]]; then
    echo "  WARNING: No .env file and no --hf-token provided."
    echo "  Some datasets may fail to download. Get a token at:"
    echo "  https://huggingface.co/settings/tokens"
    cp .env.example .env
  fi

  # Download image datasets
  echo ""
  echo "  Downloading image datasets (${MAX_SAMPLES} samples/dataset)..."
  uv run prepare.py --modality image --max-samples "${MAX_SAMPLES}"
  echo ""
  echo "  Data summary:"
  uv run prepare.py --verify --modality image
else
  echo "[2/5] Skipping setup (--skip-setup)"
  cd "${DFRESEARCH_DIR}"
  source .venv/bin/activate
fi

# ── Step 3: Train ──
echo ""
echo "[3/5] Training ${MODEL}..."
if [[ -n "${HOURS}" ]]; then
  echo "  Full training for ${HOURS} hours with ${MAX_SAMPLES} samples/dataset..."
  uv run train_full.py --modality image --model "${MODEL}" --hours "${HOURS}" --max-samples "${MAX_SAMPLES}"
else
  echo "  Quick training (10 min budget)..."
  uv run train_image.py --model "${MODEL}"
fi

# ── Step 4: Export ZIP ──
echo ""
echo "[4/5] Exporting submission ZIP..."
uv run export.py --modality image --model "${MODEL}"

CKPT_DIR="${DFRESEARCH_DIR}/results/checkpoints/image"
EXPORT_DIR="${DFRESEARCH_DIR}/results/exports"
ZIP_PATH="${EXPORT_DIR}/image_detector_${MODEL}.zip"

# ── Step 5: Summary ──
echo ""
echo "[5/5] Done!"
echo "============================================================"
echo "  Results on this GPU machine:"
echo ""
echo "  Checkpoint: ${CKPT_DIR}/"
echo "    model.safetensors"
echo "    model.py"
echo "    model_config.yaml"
echo ""
echo "  Submission ZIP: ${ZIP_PATH}"
echo ""
echo "  Next: copy the ZIP to your VPS and push:"
echo "    scp ${ZIP_PATH} your_vps:~/"
echo "    # Then on your VPS:"
echo "    gascli d push --image-model ~/image_detector_${MODEL}.zip \\"
echo "      --wallet-name <NAME> --wallet-hotkey <KEY>"
echo "============================================================"
