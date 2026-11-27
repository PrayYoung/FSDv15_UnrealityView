#!/usr/bin/env bash
set -euo pipefail
# Separate environments keep pinned Difix and the compiled trainer compatible.
bash scripts/difix_setup.sh
bash tools/chase_3dgs/setup_drivestudio.sh
export PATH="$HOME/.local/bin:$PATH"
uv cache clean
