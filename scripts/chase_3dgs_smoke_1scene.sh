#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
: "${OVERFIT_GATE:?Path to completed Stage 1A gate.json}"
: "${OVERFIT_REVIEW:?Path to hash-bound passing visual review}"
: "${FRONT_GATE:?Path to completed Stage 1B numeric report}"
SEQUENCE=${SEQUENCE:-040}
return_scene_artifacts() {
  result_code=$?
  trap - EXIT
  uv run --no-sync python -m scripts.collect_chase_return --source outputs/chase_teacher \
    --destination outputs/chase_teacher_return --sequence "$SEQUENCE" --exit-code "$result_code" || echo 'Return collection incomplete' >&2
  exit "$result_code"
}
trap return_scene_artifacts EXIT
mkdir -p outputs/chase_teacher outputs/chase_teacher_return
uv run --no-sync python -m scripts.pipeline_gates check --report "$OVERFIT_GATE" --review "$OVERFIT_REVIEW"
if [ "${1:-}" = --independent-verification ]; then
  # Explicit single-scene teacher check; never grants a front-view visual pass.
  uv run --no-sync python -m scripts.pipeline_gates check-numeric --report "$FRONT_GATE"
else
  [ "$#" = 0 ] || { echo 'Unknown smoke option' >&2; exit 2; }
  : "${FRONT_REVIEW:?Path to hash-bound passing Stage 1B visual review}"
  uv run --no-sync python -m scripts.pipeline_gates check --report "$FRONT_GATE" --review "$FRONT_REVIEW"
fi
uv run --no-sync python scripts/cuda_preflight.py
bash tools/chase_3dgs/check_build_env.sh
# Optional cleanup is explicit; the default executes raw dynamic 3DGS only.
SETUP_ARGS=();RENDER_ARGS=()
if [ "${DIFIX_CLEANUP:-0}" = 1 ]; then SETUP_ARGS+=(--difix);RENDER_ARGS+=(--difix-cleanup);fi
uv run --no-sync python -m scripts.chase_timed_stage --record outputs/chase_teacher/stage_times.json --stage build -- bash tools/chase_3dgs/setup_drivestudio.sh "${SETUP_ARGS[@]}"

# Native LiDAR/cuboids are needed here. Camera-only encoded caches cannot replace
# them. The original archive was scratch and may need remote re-download.
uv venv --python 3.11 .download-venv
uv pip install --python .download-venv/bin/python -r configs/archive_download_requirements.txt
mkdir -p outputs/chase_teacher
HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=4 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
  timeout --signal=TERM --kill-after=60s 7200 .download-venv/bin/python scripts/download_pandaset_archive.py \
  --record outputs/chase_teacher/archive_record.json
uv run --no-sync python -m tools.chase_3dgs.extract_pandaset_scene \
  --archive .cache/pandaset_archive/pandaset.zip --output .cache/chase_native --sequence "$SEQUENCE"
timeout --signal=TERM --kill-after=60s 7200 .drivestudio-venv/bin/python -m tools.chase_3dgs.render_chase_teacher \
  --native .cache/chase_native --processed .cache/chase_processed --sequence "$SEQUENCE" \
  --output outputs/chase_teacher --iters "${THREEDGS_ITERS:-15000}" "${RENDER_ARGS[@]}"
# Verify that the exact exported contract is accepted, without loading native
# pickle tables in the core package or inventing an elevated front-camera label.
uv run --no-sync python -m tools.chase_3dgs.verify_ingestion \
  --native .cache/chase_native --targets outputs/chase_teacher --sequence "$SEQUENCE" \
  --output outputs/chase_teacher/ingestion_check
