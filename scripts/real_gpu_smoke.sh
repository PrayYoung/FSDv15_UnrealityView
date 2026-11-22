#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p outputs/pandaset_smoke/verification
uv run --no-sync python scripts/cuda_preflight.py | tee outputs/pandaset_smoke/verification/cuda.json
uv run --no-sync python scripts/cuda_train_smoke.py | tee outputs/pandaset_smoke/verification/backward.txt
RUN_PRETRAINED=1 uv run --no-sync pytest -q --junitxml=outputs/pandaset_smoke/verification/pytest.xml
uv run --no-sync python scripts/fetch_pandaset_cameras.py --output .cache/pandaset_native --manifest outputs/pandaset_smoke/splits.json --frames 64 --max-gib 2
uv run --no-sync python prepare_real_data.py cache --manifest outputs/pandaset_smoke/splits.json --output .cache/real_dataset --device cuda --target front
uv run --no-sync python train_cached_meanflow.py --config configs/real_smoke.json --data .cache/real_dataset --output outputs/pandaset_smoke/train --device cuda
uv run --no-sync python evaluate.py --checkpoint outputs/pandaset_smoke/train/checkpoint --data .cache/real_dataset --output outputs/pandaset_smoke/evaluation --split val --device cuda --max-items 2
uv run --no-sync python evaluate.py --checkpoint outputs/pandaset_smoke/train/checkpoint --data .cache/real_dataset --output outputs/pandaset_smoke/test --split test --device cuda --max-items 2
uv run --no-sync python benchmark_edge.py --checkpoint outputs/pandaset_smoke/train/checkpoint --output outputs/pandaset_smoke/benchmark.json --device cuda --repeats 3
uv run --no-sync python scripts/export_real_smoke.py --data .cache/real_dataset --output outputs/pandaset_smoke
uv run --no-sync python scripts/verify_artifacts.py outputs/pandaset_smoke
