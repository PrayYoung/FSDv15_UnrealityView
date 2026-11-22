#!/usr/bin/env bash
set -euo pipefail
timeout --signal=TERM --kill-after=60s 2700 bash scripts/real_gpu_smoke.sh
