#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" "/var/log/portal/opd32b-eval.log"
. "${utils}/environment.sh"

cd /workspace/aimo-proof-pilot-eval
export CUDA_VISIBLE_DEVICES=0,1
export VENV=/workspace/pp/venv
exec /bin/bash serve_opd32b.sh \
  --config evaluation/configs/nemotron_cascade2.yaml
