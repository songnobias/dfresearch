#!/usr/bin/env bash
# Sync dfresearch venv only. Use this on machines where global uv/pip sets
# UV_EXTRA_INDEX_URL to PyTorch CUDA (causes markupsafe to resolve from cu128 → cp314 error).
#
# Run: ./scripts/sync_env.sh   (from dfresearch root)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -f "${ROOT}/pyproject.toml" ]]; then
  echo "Expected pyproject.toml in ${ROOT}" >&2
  exit 1
fi

# Parent "gas" project / global CUDA index — must not apply to dfresearch resolution
unset VIRTUAL_ENV UV_PROJECT || true
unset UV_EXTRA_INDEX_URL UV_INDEX_URL PIP_EXTRA_INDEX_URL || true

export UV_PROJECT="${ROOT}"

echo "UV_PROJECT=${UV_PROJECT}"
echo "Syncing dfresearch only (not Bitmind root)..."
rm -rf "${ROOT}/.venv"
uv sync "$@"

echo "Done. Activate: source ${ROOT}/.venv/bin/activate"
