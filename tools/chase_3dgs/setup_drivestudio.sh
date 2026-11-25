#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
source tools/chase_3dgs/cuda_env.sh
bash tools/chase_3dgs/check_build_env.sh
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
  timeout 600 apt-get update -qq
  DEBIAN_FRONTEND=noninteractive timeout 600 apt-get install -y -qq build-essential git python3.10-dev libgl1-mesa-dev libegl1-mesa-dev
fi
bash tools/chase_3dgs/compile_cuda_probe.sh
uv venv --python 3.10 .drivestudio-venv
PY="$ROOT/.drivestudio-venv/bin/python"
"$PY" tools/chase_3dgs/build_env.py --check-headers
export MAX_JOBS=$("$PY" tools/chase_3dgs/build_env.py --jobs)
uv pip install --python "$PY" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$PY" -r tools/chase_3dgs/requirements.txt
"$PY" tools/chase_3dgs/setup_upstream.py "$@"
for package in gsplat pytorch3d nvdiffrast; do
  uv pip install --python "$PY" --no-build-isolation --no-deps ".cache/third_party/$package"
done
uv pip install --python "$PY" --no-build-isolation --no-deps .cache/third_party/pandaset/python
PYTHONPATH="$ROOT/.cache/third_party/drivestudio:$ROOT/src:$ROOT" "$PY" -c 'import torch; from models.trainers import MultiTrainer; from models.nodes import RigidNodes,DeformableNodes; assert torch.cuda.is_available(); print("DYNAMIC_3DGS_IMPORT_CUDA_OK")'
"$PY" tools/chase_3dgs/cuda_build_preflight.py

if [ -d .cache/third_party/difix ]; then
  uv pip install --python "$PY" diffusers==0.33.1 accelerate==1.4.0 peft==0.14.0
fi
