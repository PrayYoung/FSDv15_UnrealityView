#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=2
export MAX_JOBS=2
mkdir -p outputs/upgrade_check
# All stages run under one managed lease; bound the entire verification workload.
timeout --signal=TERM --kill-after=60s 1800 bash scripts/upgrade_gpu_checks.sh
