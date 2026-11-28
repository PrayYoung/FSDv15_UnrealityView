#!/usr/bin/env bash
set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then
  curl --fail --location --silent --show-error https://astral.sh/uv/install.sh -o /tmp/unrealityview-uv-install.sh
  sh /tmp/unrealityview-uv-install.sh
fi
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
source scripts/difix_cuda_env.sh
.venv/bin/python scripts/cuda_preflight.py
uv pip install --python .venv/bin/python -r configs/depth_fast_requirements.txt
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -c 'import json;from pathlib import Path;from tools.chase_3dgs.setup_upstream import checkout;p=json.loads(Path("configs/depth_metric_pins.json").read_text());checkout(Path(".cache/third_party"),"depth_metric",p["source"])'
PYTHONPATH="$PWD/.cache/third_party/depth_metric/metric_depth" .venv/bin/python -c 'from depth_anything_v2.dpt import DepthAnythingV2;print("DEPTH_METRIC_IMPORT_OK")'
.venv/bin/python scripts/depth_fp16_preflight.py
uv cache clean
