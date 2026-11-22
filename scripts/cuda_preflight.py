import json
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA_PREFLIGHT_FAILED: CUDA is unavailable")
x=torch.randn(512,512,device="cuda")
y=x@x
torch.cuda.synchronize()
if not torch.isfinite(y).all():raise SystemExit("CUDA_PREFLIGHT_FAILED: nonfinite matmul")
print(json.dumps({"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
                  "device":torch.cuda.get_device_name(0),"bf16":torch.cuda.is_bf16_supported(),
                  "status":"CUDA_PREFLIGHT_OK"}),flush=True)
