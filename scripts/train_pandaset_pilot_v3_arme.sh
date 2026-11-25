#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=2
OUT=outputs/pandaset_pilot_v3_arme
mkdir -p "$OUT"
return_artifacts() {
  result_code=$?
  trap - EXIT
  if [ -f "$OUT/train/checkpoint/training_state.json" ]; then
    uv run --no-sync python -m scripts.compact_return_artifacts --train "$OUT/train" --compact-visuals || echo 'Compaction incomplete; retained unverified files' >&2
  fi
  exit "$result_code"
}
trap return_artifacts EXIT
uv run --no-sync python scripts/cuda_preflight.py
uv run --no-sync python -m scripts.validate_pilot_cache --data recovery_cache --training-only --training-inputs-only
# Part 1 completed locally. Verify its one-episode fingerprint without rerunning
# DINO, re-encoding, modifying the parity limit or accessing reserved test data.
uv run --no-sync python -m scripts.verify_local_parity_report --data recovery_cache --report inputs/local_parity_summary.json --output "$OUT/local_parity_reference.json"
timeout --signal=TERM --kill-after=60s 6000 uv run --no-sync python train_cached_meanflow.py \
  --config configs/pandaset_pilot_v3_arme.json --data recovery_cache \
  --output "$OUT/train" --device cuda --stop-after 5000 --evaluate-initial
uv run --no-sync python -m scripts.pipeline_gates front-report --train "$OUT/train" --output "$OUT/gate.json"
uv run --no-sync python -m scripts.uve_grid_reference --train "$OUT/train" --output "$OUT/best_reference.json"
uv run --no-sync python -m scripts.diagnose_base_blur --checkpoint "$OUT/train/checkpoint_best" --data recovery_cache \
  --reference-metrics "$OUT/best_reference.json" --output "$OUT/diagnosis" --device cuda
uv run --no-sync python -m scripts.report_arm_e --directory "$OUT" --baseline inputs/baseline_diagnosis.json
