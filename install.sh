#!/usr/bin/env bash
set -Eeuo pipefail

# Install Proof Pilot directly on a Vast.ai base-image instance. This mirrors
# the Docker image bootstrap because Docker-in-Docker is unavailable there.

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$REPO/config.yaml}"
COMMAND="${1:-bootstrap}"
UV_PYTHON="${UV_PYTHON:-/usr/bin/python3}"
UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-/root/.local/bin}"

export REPO CONFIG UV_TOOL_BIN_DIR
export PATH="$UV_TOOL_BIN_DIR:$PATH"

die() {
    printf '[install] ERROR: %s\n' "$*" >&2
    exit 1
}

command -v uv >/dev/null || die "uv is required (the Vast base image includes it)"
[[ -x "$UV_PYTHON" ]] || die "Python interpreter is unavailable: $UV_PYTHON"
[[ -f "$CONFIG" ]] || die "configuration does not exist: $CONFIG"

if ! command -v kaggle >/dev/null; then
    printf '[install] installing Kaggle CLI 2.2.3\n'
    uv tool install --python "$UV_PYTHON" 'kaggle==2.2.3'
fi

if ! command -v hf >/dev/null; then
    printf '[install] installing Hugging Face Hub CLI 1.18.0\n'
    uv tool install --python "$UV_PYTHON" 'huggingface-hub==1.18.0'
fi

printf '[install] repo=%s config=%s command=%s\n' "$REPO" "$CONFIG" "$COMMAND"
exec bash "$REPO/docker/entrypoint.sh" "$COMMAND"
