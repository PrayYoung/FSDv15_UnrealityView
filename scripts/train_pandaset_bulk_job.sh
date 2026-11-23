#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=2
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
mkdir -p outputs/pandaset_bulk
timeout --signal=TERM --kill-after=60s 14400 bash scripts/train_pandaset_bulk.sh
