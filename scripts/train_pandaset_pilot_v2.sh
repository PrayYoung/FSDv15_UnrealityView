#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=2
: "${OVERFIT_GATE:?Path to completed Stage 1A gate.json}"
: "${OVERFIT_REVIEW:?Path to hash-bound passing visual review}"
uv run --no-sync python -m scripts.pipeline_gates check --report "$OVERFIT_GATE" --review "$OVERFIT_REVIEW"
# Stage 1B consumes the full preserved 60/8/8 cache, not the tiny overfit bundle.
uv run --no-sync python -m scripts.validate_pilot_cache --data recovery_cache
uv run --no-sync python scripts/cuda_preflight.py
mkdir -p outputs/pandaset_pilot_v2
timeout --signal=TERM --kill-after=60s 14400 uv run --no-sync python train_cached_meanflow.py \
  --config configs/pandaset_pilot_v2.json --data recovery_cache \
  --output outputs/pandaset_pilot_v2/train --device cuda --stop-after 5000 --evaluate-initial
uv run --no-sync python -m scripts.pipeline_gates front-report \
  --train outputs/pandaset_pilot_v2/train --output outputs/pandaset_pilot_v2/gate.json
