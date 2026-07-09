#!/bin/bash
# serve_opd32b.sh — serve the bf16 opd-32b-deploy (Olmo3Sink) with the
# proof-pilot patched sglang on an H200.
#
# Wraps proof-pilot's serve_final.sh with the deltas vs its Kaggle defaults:
#   - CONFIG=w4a16 passes no --quantization flag; pointing GPTQ at the bf16
#     model dir just serves bf16 (sglang reads dtype from the config).
#   - FLASHINFER_CUDA_ARCH_LIST=9.0a: H200 is sm90, not Kaggle's sm120.
#   - VENV is the relocated proof-pilot-env venv (see README for setup).
#
# Usage:  bash serve_opd32b.sh            # GPU 0, port 30000
#         PORT=30001 CUDA_VISIBLE_DEVICES=1 bash serve_opd32b.sh
set -euo pipefail

export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/proof-pilot" && pwd)}"
export VENV="${VENV:-/workspace/pp/venv}"
export CONFIG=w4a16
export GPTQ="${MODEL:-/workspace/models/opd-32b-deploy}"
export FLASHINFER_CUDA_ARCH_LIST=9.0a
export PORT="${PORT:-30000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MEMFRAC="${MEMFRAC:-0.88}"
export CTX="${CTX:-200000}"

exec bash "$REPO/kaggle_deploy/final/serve/serve_final.sh"
