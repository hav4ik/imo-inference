#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-/workspace/pp/venv}"

uv pip install \
  --python "$VENV/bin/python" \
  --requirement "$REPO/evaluation/requirements.txt"

uv pip install \
  --reinstall \
  --no-deps \
  --python "$VENV/bin/python" \
  'nvidia-cutlass-dsl==4.5.2' \
  'nvidia-cutlass-dsl-libs-base==4.5.2' \
  'nvidia-cutlass-dsl-libs-cu13==4.5.2'
