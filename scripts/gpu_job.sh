#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Upper bound on workload wall time; inherited runner hourly cap still applies.
# The runner owns output retrieval and destruction even if timeout terminates us.
timeout --signal=TERM --kill-after=60s 5400 bash scripts/gpu_pipeline.sh
