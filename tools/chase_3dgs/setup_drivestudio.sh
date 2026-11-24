#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
bash tools/chase_3dgs/check_build_env.sh
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
  timeout 600 apt-get update -qq
  DEBIAN_FRONTEND=noninteractive timeout 600 apt-get install -y -qq build-essential git libgl1-mesa-dev libegl1-mesa-dev
fi
uv venv --python 3.10 .drivestudio-venv
PY="$ROOT/.drivestudio-venv/bin/python"
uv pip install --python "$PY" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$PY" -r tools/chase_3dgs/requirements.txt
"$PY" tools/chase_3dgs/setup_upstream.py "$@"
export MAX_JOBS=4
for package in gsplat pytorch3d nvdiffrast; do
  uv pip install --python "$PY" --no-build-isolation --no-deps ".cache/third_party/$package"
done
uv pip install --python "$PY" --no-build-isolation --no-deps .cache/third_party/pandaset/python
PYTHONPATH="$ROOT/.cache/third_party/drivestudio:$ROOT/src:$ROOT" "$PY" -c 'import torch; from models.trainers import MultiTrainer; from models.nodes import RigidNodes,DeformableNodes; assert torch.cuda.is_available(); print("DYNAMIC_3DGS_IMPORT_CUDA_OK")'

if [ -d .cache/third_party/difix ]; then
  uv pip install --python "$PY" diffusers==0.33.1 accelerate==1.4.0 peft==0.14.0
fi
