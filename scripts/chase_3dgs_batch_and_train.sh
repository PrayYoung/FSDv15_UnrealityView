#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
: "${FRONT_GATE:?Stage 1B numeric report}"
: "${FRONT_REVIEW:?Stage 1B hash-bound visual review}"
: "${TEACHER_GATE:?Stage 2A quality.json}"
: "${TEACHER_REVIEW:?Stage 2A hash-bound visual review}"
: "${FRONT_CHECKPOINT:?Stage 1B best weights}"
uv run --no-sync python -m scripts.pipeline_gates check --report "$FRONT_GATE" --review "$FRONT_REVIEW"
uv run --no-sync python -m scripts.pipeline_gates check --report "$TEACHER_GATE" --review "$TEACHER_REVIEW"
uv run --no-sync python scripts/cuda_preflight.py
# Requires the checksum-pinned archive on remote storage. Setup/download are
# explicit preparation, not silently repeated inside a 76-scene batch.
case "${1:-render}" in
render)
.drivestudio-venv/bin/python -m tools.chase_3dgs.batch \
  --archive .cache/pandaset_archive/pandaset.zip --front-gate "$FRONT_GATE" --front-review "$FRONT_REVIEW" \
  --teacher-gate "$TEACHER_GATE" --teacher-review "$TEACHER_REVIEW" --iters "${THREEDGS_ITERS:-15000}"
;;
cache)
uv run --no-sync python scripts/fetch_pandaset_cameras.py \
  --local-archive .cache/pandaset_archive/pandaset.zip --output .cache/chase_cameras \
  --manifest outputs/chase_teacher/native_splits.json --selection configs/pandaset_validated_pilot.json \
  --train-count 60 --val-count 8 --test-count 8 --frames 80 --max-gib 20 --workers 2
uv run --no-sync python prepare_real_data.py cache --target teacher --targets outputs/chase_teacher \
  --manifest outputs/chase_teacher/native_splits.json --output outputs/chase_base/data --device cuda
;;
train)
uv run --no-sync python train_cached_meanflow.py --config configs/pandaset_pilot_v2.json \
  --data outputs/chase_base/data --init-weights "$FRONT_CHECKPOINT" \
  --output outputs/chase_base/train --device cuda --stop-after 5000 --evaluate-initial
;;
*) echo "Choose one stage: render, cache, or train" >&2; exit 2;;
esac
