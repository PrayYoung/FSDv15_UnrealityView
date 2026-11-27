#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
source scripts/difix_cuda_env.sh
.venv/bin/python scripts/cuda_preflight.py
# Only existing raw renders and front references are uploaded; no native archive.
timeout --signal=TERM --kill-after=60s 3600 .venv/bin/python -m tools.chase_3dgs.cleanup_existing \
  --inputs inputs/scene040 --output outputs/difix_return
.venv/bin/python scripts/seal_return.py outputs/difix_return
