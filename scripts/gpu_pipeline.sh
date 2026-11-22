#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p outputs/verification
uv run --no-sync python scripts/cuda_preflight.py | tee outputs/verification/cuda_preflight.json
uv run --no-sync python scripts/cuda_train_smoke.py | tee outputs/verification/cuda_train_smoke.txt
RUN_PRETRAINED=1 uv run --no-sync pytest -q --junitxml=outputs/verification/pytest.xml | tee outputs/verification/pytest.txt
uv run --no-sync python collect_metadrive.py --device cuda --output outputs/dataset
uv run --no-sync python train_cached_meanflow.py --device cuda --config configs/train.json --data outputs/dataset --output outputs/train
uv run --no-sync python evaluate.py --device cuda --data outputs/dataset --checkpoint outputs/train/checkpoint --output outputs/evaluation
uv run --no-sync python benchmark_edge.py --device cuda --checkpoint outputs/train/checkpoint --output outputs/benchmark.json
uv run --no-sync python scripts/verify_artifacts.py outputs
