"""Fail on incomplete/corrupt local or retrieved evidence; never unpickle weights."""
import json
from pathlib import Path
import sys
from PIL import Image
from safetensors import safe_open

root=Path(sys.argv[1] if len(sys.argv)>1 else "outputs")
checkpoint=root/"train/checkpoint/model.safetensors"
with safe_open(checkpoint,framework="pt",device="cpu") as f:
    keys=list(f.keys())
    assert any(k.startswith("dit.") for k in keys) and any(k.startswith("policy.") for k in keys)
    for key in keys:
        tensor=f.get_tensor(key)
        assert tensor.isfinite().all(),key
for path in root.rglob("*.jsonl"):
    lines=path.read_text().splitlines();assert lines,path
    for line in lines:json.loads(line)
for path in root.rglob("*.json"):json.loads(path.read_text())
for path in root.rglob("*.png"):
    with Image.open(path) as image:image.verify()
for path in root.rglob("*.gif"):
    with Image.open(path) as image:
        assert image.n_frames>=32,path
        image.seek(image.n_frames-1);image.load()
assert (root/"evaluation/metrics.json").is_file()
assert (root/"benchmark.json").is_file()
print("ARTIFACTS_VERIFIED",flush=True)
