#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=4 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export UNREALITY_PROGRESS_FILE="$PWD/outputs/scene040_diagnostic_return/optimizer_progress.json"
RETURN=outputs/scene040_diagnostic_return
mkdir -p "$RETURN"
cp SOURCE_MANIFEST.json "$RETURN/source_manifest.json"
finish() {
  code=$?;trap - EXIT
  .venv/bin/python scripts/seal_return.py "$RETURN" || true
  exit "$code"
}
trap finish EXIT
source scripts/difix_cuda_env.sh
.venv/bin/python scripts/cuda_preflight.py
.venv/bin/python -m tools.chase_3dgs.prepare_native --output "$RETURN"
source tools/chase_3dgs/cuda_env.sh
timeout --signal=TERM --kill-after=60s 6600 .drivestudio-venv/bin/python -m tools.chase_3dgs.diagnostic_refit \
  --native .cache/chase_native --processed .cache/chase_processed --scratch .cache/scene040_diagnostic \
  --work .cache/scene040_diagnostic_views --output "$RETURN"
source scripts/difix_cuda_env.sh
timeout --signal=TERM --kill-after=60s 1800 .venv/bin/python -m tools.chase_3dgs.cleanup_diagnostic --work .cache/scene040_diagnostic_views --output "$RETURN"
