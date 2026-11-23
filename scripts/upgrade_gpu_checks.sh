#!/usr/bin/env bash
set -euo pipefail
command -v nvcc >/dev/null || { echo "CUDA development toolkit (nvcc) required for the optional gsplat check"; exit 1; }
uv sync --locked --extra dev --extra realdata --extra teacher
uv run --no-sync python scripts/cuda_preflight.py
RUN_PRETRAINED=1 uv run --no-sync pytest -q --tb=short
uv run --no-sync python scripts/verify_upgrade_cuda.py
uv run --no-sync python scripts/fetch_pandaset_cameras.py --output .cache/pandaset_native --manifest .cache/upgrade_splits.json --frames 64 --max-gib 2
uv run --no-sync python prepare_real_data.py cache --manifest .cache/upgrade_splits.json --output .cache/upgrade_data --device cuda --target front
uv run --no-sync python train_cached_meanflow.py --config configs/real_smoke.json --data .cache/upgrade_data --output outputs/upgrade_check/base --device cuda --steps 4 --stop-after 2
uv run --no-sync python train_cached_meanflow.py --config configs/real_smoke.json --data .cache/upgrade_data --output outputs/upgrade_check/base --device cuda --steps 4 --resume outputs/upgrade_check/base/checkpoint
uv run --no-sync python prepare_style_cache.py --data .cache/upgrade_data --output .cache/upgrade_styles --device cuda
for style in comic cyberpunk; do
  uv run --no-sync python train_style_lora.py --base outputs/upgrade_check/base/checkpoint --data .cache/upgrade_data --styles-data .cache/upgrade_styles --style "$style" --output "outputs/upgrade_check/adapters/$style" --device cuda --steps 2 --stop-after 1 --batch-size 2 --grad-accum 1 --eval-items 1
  uv run --no-sync python train_style_lora.py --base outputs/upgrade_check/base/checkpoint --data .cache/upgrade_data --styles-data .cache/upgrade_styles --style "$style" --output "outputs/upgrade_check/adapters/$style" --device cuda --steps 2 --resume "outputs/upgrade_check/adapters/$style/checkpoint" --batch-size 2 --grad-accum 1 --eval-items 1
done
uv run --no-sync python benchmark_styles.py --base outputs/upgrade_check/base/checkpoint --lora-dir outputs/upgrade_check/adapters --device cuda --output outputs/upgrade_check/style_benchmark.json
