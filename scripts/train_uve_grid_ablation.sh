#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=2
OUT=outputs/uve10x16
mkdir -p "$OUT"
return_artifacts() {
  result_code=$?
  trap - EXIT
  if [ -f "$OUT/train/checkpoint/training_state.json" ]; then
    uv run --no-sync python -m scripts.compact_return_artifacts --train "$OUT/train" --compact-visuals || echo 'Compaction incomplete; unverified files preserved' >&2
  fi
  exit "$result_code"
}
trap return_artifacts EXIT
uv run --no-sync python scripts/cuda_preflight.py
uv run --no-sync python -m scripts.validate_pilot_cache --data recovery_cache
uv run --no-sync python -m scripts.reencode_uve_cache --source recovery_cache --output recovery_cache_uve10x16 --grid 10 16 --device cuda --parity-report "$OUT/parity.json"
BATCH=16;ACCUM=1
set +e
uv run --no-sync python -m scripts.uve_grid_preflight --data recovery_cache_uve10x16 --config configs/pandaset_pilot_v2.json --batch-size 16 --grad-accum 1 --output "$OUT/preflight_batch16.json"
PREFLIGHT_STATUS=$?
set -e
if [ "$PREFLIGHT_STATUS" = 42 ]; then
  BATCH=8;ACCUM=2
  uv run --no-sync python -m scripts.uve_grid_preflight --data recovery_cache_uve10x16 --config configs/pandaset_pilot_v2.json --batch-size 8 --grad-accum 2 --output "$OUT/preflight_batch8.json"
elif [ "$PREFLIGHT_STATUS" != 0 ]; then
  exit "$PREFLIGHT_STATUS"
fi
uv run --no-sync python -m scripts.uve_grid_budget --directory "$OUT"
timeout --signal=TERM --kill-after=60s 14400 uv run --no-sync python train_cached_meanflow.py \
  --config configs/pandaset_pilot_v2.json --data recovery_cache_uve10x16 --output "$OUT/train" --device cuda \
  --batch-size "$BATCH" --grad-accum "$ACCUM" --stop-after 5000 --evaluate-initial
uv run --no-sync python -m scripts.pipeline_gates front-report --train "$OUT/train" --output "$OUT/gate.json"
uv run --no-sync python -m scripts.uve_grid_reference --train "$OUT/train" --output "$OUT/best_reference.json"
uv run --no-sync python -m scripts.diagnose_base_blur --checkpoint "$OUT/train/checkpoint_best" --data recovery_cache_uve10x16 \
  --reference-metrics "$OUT/best_reference.json" --output "$OUT/diagnosis" --device cuda
uv run --no-sync python -m scripts.report_uve_ablation --directory "$OUT" --baseline inputs/baseline_diagnosis.json
