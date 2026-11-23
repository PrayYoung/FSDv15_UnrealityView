#!/usr/bin/env bash
set -euo pipefail
OUT=outputs/pandaset_bulk
uv run --no-sync python scripts/cuda_preflight.py
uv run --no-sync python scripts/cuda_train_smoke.py
uv venv --python 3.11 .download-venv
uv pip install --python .download-venv/bin/python -r configs/archive_download_requirements.txt
timeout --signal=TERM --kill-after=30s 7200 .download-venv/bin/python scripts/download_pandaset_archive.py \
  --record "$OUT/archive_record.json"
uv run --no-sync python scripts/fetch_pandaset_cameras.py \
  --output .cache/pandaset_native --manifest "$OUT/splits.json" \
  --selection inputs/preflight.json --recovery-cache recovery_cache \
  --train-count 60 --val-count 8 --test-count 8 --frames 80 --max-gib 20 --workers 2 \
  --local-archive .cache/pandaset_archive/pandaset.zip
uv run --no-sync python scripts/seed_panda_recovery.py \
  --recovery recovery_cache --manifest "$OUT/splits.json" --output "$OUT/data"
uv run --no-sync python prepare_real_data.py cache \
  --manifest "$OUT/splits.json" --output "$OUT/data" --device cuda --target front --reuse-existing
uv run --no-sync python train_cached_meanflow.py \
  --config configs/pandaset_pilot.json --data "$OUT/data" --output "$OUT/train" \
  --device cuda --stop-after 5000 --evaluate-initial
