#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare a fresh Vast.ai container for GitHub and Codex-driven installation.
# Codex prompt: please see the repo workspace/aimo-proof-pilot-inference and run the installation

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_CODEX="${LAUNCH_CODEX:-1}"

die() {
    printf '[container-setup] ERROR: %s\n' "$*" >&2
    exit 1
}

install_github_cli() {
    command -v gh >/dev/null && return

    printf '[container-setup] installing GitHub CLI\n'
    if [[ "$(id -u)" -eq 0 ]]; then
        apt-get update
        apt-get install -y gh
    elif command -v sudo >/dev/null; then
        sudo apt-get update
        sudo apt-get install -y gh
    else
        die "installing gh requires root or passwordless sudo"
    fi
}

load_node() {
    command -v npm >/dev/null && return
    [[ -s /opt/nvm/nvm.sh ]] || die "npm is unavailable and /opt/nvm/nvm.sh does not exist"
    # shellcheck disable=SC1091
    source /opt/nvm/nvm.sh
}

authenticate_github() {
    if gh auth status --hostname github.com >/dev/null 2>&1; then
        printf '[container-setup] GitHub authentication is already active\n'
    elif [[ -n "${GH_TOKEN:-}" ]]; then
        gh auth status --hostname github.com >/dev/null \
            || die "GH_TOKEN was provided but GitHub rejected it"
        printf '[container-setup] using GitHub authentication from GH_TOKEN\n'
    elif [[ -t 0 && -t 1 ]]; then
        printf '%s\n' \
            '[container-setup] complete the device login at https://github.com/login/device'
        GH_BROWSER="${GH_BROWSER:-echo}" gh auth login \
            --hostname github.com \
            --git-protocol https \
            --web \
            --scopes repo,workflow
    else
        die "GitHub is not authenticated; set GH_TOKEN or run this script interactively"
    fi

    gh auth setup-git --hostname github.com
}

install_github_cli
load_node

printf '[container-setup] installing the latest OpenAI Codex CLI\n'
npm install -g @openai/codex

authenticate_github

printf '[container-setup] setup complete; repository=%s\n' "$REPO"
if [[ "$LAUNCH_CODEX" == "1" ]]; then
    cd "$REPO"
    exec codex
fi
