#!/bin/bash
set -euo pipefail

run_root=/workspace/aimo-proof-pilot-eval/evaluation/runs/imo-2025-problem-1-dryrun-20260712
mkdir -p "${run_root}"

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" "${run_root}/runner.log"
. "${utils}/environment.sh"

cd /workspace/aimo-proof-pilot-eval
exec /workspace/pp/venv/bin/python -u \
  evaluation/harness/run_full_evaluation.py \
  --config evaluation/configs/nemotron_cascade2.yaml \
  --ids-file evaluation/manifests/imo-2025-problem-1.json \
  --run-id imo-2025-problem-1-dryrun-20260712
