#!/usr/bin/env bash
set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then
  curl --fail --location --silent --show-error https://astral.sh/uv/install.sh -o /tmp/unrealityview-uv-install.sh
  sh /tmp/unrealityview-uv-install.sh
fi
export PATH="$HOME/.local/bin:$PATH"
timeout --signal=TERM --kill-after=30s 1200 bash -c '
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
# Check the actual GPU before fetching Difix dependencies or model weights.
source scripts/difix_cuda_env.sh
.venv/bin/python scripts/cuda_preflight.py
uv pip install --python .venv/bin/python -r configs/difix_requirements.txt
.venv/bin/python scripts/setup_difix_runtime.py
PYTHONPATH="$PWD/.cache/third_party/difix/src" .venv/bin/python -c "from pipeline_difix import DifixPipeline; print(\"DIFIX_PIPELINE_IMPORT_OK\")"
'
