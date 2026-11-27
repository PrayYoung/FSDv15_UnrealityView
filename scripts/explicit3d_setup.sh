#!/usr/bin/env bash
set -euo pipefail
bash scripts/difix_setup.sh
export PATH="$HOME/.local/bin:$PATH"
uv pip install --python .venv/bin/python --constraint configs/difix_requirements.txt pandas==2.2.3 opencv-python-headless==4.11.0.86
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -c 'import json;from pathlib import Path;from tools.chase_3dgs.setup_upstream import checkout;p=json.loads(Path("configs/depth_metric_pins.json").read_text());checkout(Path(".cache/third_party"),"depth_metric",p["source"])'
PYTHONPATH="$PWD/.cache/third_party/depth_metric/metric_depth" .venv/bin/python -c 'from depth_anything_v2.dpt import DepthAnythingV2;print("DEPTH_METRIC_IMPORT_OK")'
uv cache clean
