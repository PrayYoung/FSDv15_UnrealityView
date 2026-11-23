#!/usr/bin/env bash
set -euo pipefail
OUT=outputs/pandaset_pilot
uv run --no-sync python scripts/cuda_preflight.py
uv run --no-sync python scripts/cuda_train_smoke.py
uv run --no-sync python scripts/fetch_pandaset_cameras.py \
  --output .cache/pandaset_native --manifest "$OUT/splits.json" \
  --train-count 60 --val-count 8 --test-count 8 --frames 80 --max-gib 20
# Retain the encoded, resized cache for exact resume after this lease ends.
# Original full-resolution native cameras remain only on remote scratch.
uv run --no-sync python prepare_real_data.py cache \
  --manifest "$OUT/splits.json" --output "$OUT/data" --device cuda --target front
uv run --no-sync python train_cached_meanflow.py \
  --config configs/pandaset_pilot.json --data "$OUT/data" --output "$OUT/train" \
  --device cuda --stop-after 5000 --evaluate-initial
