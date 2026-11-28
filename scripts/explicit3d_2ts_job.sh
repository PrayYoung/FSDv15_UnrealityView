#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=4 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
RETURN=outputs/explicit3d_2ts_return
mkdir -p "$RETURN"
cp SOURCE_MANIFEST.json "$RETURN/source_manifest.json"
cp .cache/depth_fp16_preflight.json "$RETURN/depth_fp16_preflight.json"
finish() {
  code=$?;trap - EXIT
  .venv/bin/python scripts/seal_return.py "$RETURN" || true
  exit "$code"
}
trap finish EXIT
source scripts/difix_cuda_env.sh
.venv/bin/python scripts/cuda_preflight.py
.venv/bin/python -m tools.chase_3dgs.prepare_native --output "$RETURN"
timeout --signal=TERM --kill-after=60s 1800 .venv/bin/python -m tools.chase_3dgs.explicit_two_timestamp \
  --native .cache/chase_native --teacher inputs/teacher --output "$RETURN"
