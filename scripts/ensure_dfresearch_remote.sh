#!/usr/bin/env bash
# Fix origin when dfresearch was cloned with the wrong remote (e.g. Bitmind umbrella URL).
# Run from dfresearch repo root: ./scripts/ensure_dfresearch_remote.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if [[ ! -f "${ROOT}/train_image.py" ]]; then
  echo "Run this inside the dfresearch directory (train_image.py not found)." >&2
  exit 1
fi

# Override if you use a fork: export DFRESEARCH_ORIGIN_URL='https://github.com/you/dfresearch.git'
TARGET="${DFRESEARCH_ORIGIN_URL:-https://github.com/songnobias/dfresearch.git}"
CURRENT="$(git remote get-url origin 2>/dev/null || true)"

if [[ "${CURRENT}" == "${TARGET}" ]]; then
  echo "origin is already: ${TARGET}"
  exit 0
fi

echo "Fixing origin:"
echo "  from: ${CURRENT:-<none>}"
echo "  to:   ${TARGET}"
git remote set-url origin "${TARGET}"
git remote -v
