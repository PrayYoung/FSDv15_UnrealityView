#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
export UNREALITY_PROGRESS_FILE="$PWD/outputs/pixart_stateless_v2_return/optimizer_progress.json"
uv run --no-sync python scripts/cuda_preflight.py
# Native-camera pilot only; this script never downloads the PandaSet archive.
timeout --signal=TERM --kill-after=60s 7200 uv run --no-sync python scripts/train_pixart_stateless.py \
  --cache inputs/recovery_cache --arm-e inputs/arm_e_best --v1-adapter inputs/v1_adapter.safetensors \
  --output outputs/pixart_stateless_v2_return --steps 2000 --batch-size 8 --grad-accum 2
