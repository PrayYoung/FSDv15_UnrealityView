#!/usr/bin/env bash
set -euo pipefail
bash scripts/scene040_diagnostic_setup.sh
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
source scripts/difix_cuda_env.sh
uv pip install --python .venv/bin/python --constraint configs/difix_requirements.txt opencv-python-headless==4.11.0.86
.venv/bin/python -c 'import json;from pathlib import Path;from tools.chase_3dgs.setup_upstream import checkout;p=json.loads(Path("configs/depth_metric_pins.json").read_text());checkout(Path(".cache/third_party"),"depth_metric",p["source"])'
.venv/bin/python scripts/depth_fp16_preflight.py
.venv/bin/python scripts/flying_cache_preflight.py
uv cache clean
