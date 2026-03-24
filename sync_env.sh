#!/usr/bin/env bash
# Same as scripts/sync_env.sh — kept at repo root so you can run: ./sync_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
[[ -f "${ROOT}/pyproject.toml" ]] || { echo "No pyproject.toml in ${ROOT}" >&2; exit 1; }
unset VIRTUAL_ENV UV_PROJECT || true
unset UV_EXTRA_INDEX_URL UV_INDEX_URL PIP_EXTRA_INDEX_URL || true
export UV_PROJECT="${ROOT}"
echo "UV_PROJECT=${UV_PROJECT}"
rm -rf "${ROOT}/.venv"
uv sync "$@"
echo "Done: source ${ROOT}/.venv/bin/activate"
