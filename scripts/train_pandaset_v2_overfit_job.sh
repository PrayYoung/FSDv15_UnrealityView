#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=2
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p outputs/pandaset_v2_overfit
uv run --no-sync python scripts/cuda_preflight.py
uv run --no-sync python scripts/cuda_train_smoke.py
timeout --signal=TERM --kill-after=60s 1800 uv run --no-sync python -m scripts.check_pandaset_v2_overfit \
  --data recovery_cache --output outputs/pandaset_v2_overfit --device cuda
