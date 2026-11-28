#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=4 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export WANDB_MODE=disabled WANDB_SILENT=true
RETURN=outputs/flying_cache_return
mkdir -p "$RETURN"
cp SOURCE_MANIFEST.json "$RETURN/source_manifest.json"
cp .cache/flying_cache_preflight.json "$RETURN/cache_preflight.json"
finish() {
  code=$?;trap - EXIT
  .venv/bin/python scripts/seal_return.py "$RETURN" || true
  exit "$code"
}
trap finish EXIT
source scripts/difix_cuda_env.sh
.venv/bin/python scripts/cuda_preflight.py
CHUNK=$(.venv/bin/python -c 'import json;print(json.load(open("inputs/chunk.json"))["id"])')
timeout --signal=TERM --kill-after=60s 18000 .venv/bin/python -m tools.chase_3dgs.flying_chunk --chunk "$CHUNK" --output "$RETURN"
