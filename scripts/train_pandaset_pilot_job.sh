#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=2
mkdir -p outputs/pandaset_pilot
# Bound spending even on an unexpectedly slow host. The runner owns cleanup.
timeout --signal=TERM --kill-after=60s 10800 bash scripts/train_pandaset_pilot.sh
