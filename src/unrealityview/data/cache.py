import json
import os
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def resolve_safe_subpath(root, subpath):
    root = Path(root).resolve()
    value = Path(subpath)
    if value.is_absolute():
        raise ValueError("Absolute paths are not allowed")
    path = (root/value).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Path escapes the configured data root")
    return path


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,path)


def save_episode(directory, tensors, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    validate_episode(tensors)
    metadata={**metadata,"uve_grid":list(episode_grid(tensors,metadata))}
    temporary = directory/"episode.safetensors.tmp"
    save_file({k:v.detach().cpu().contiguous() for k,v in tensors.items()},str(temporary),
              metadata={"format":"unrealityview-v2","fps":str(metadata.get("fps",36))})
    os.replace(temporary,directory/"episode.safetensors")
    atomic_json(directory/"metadata.json",metadata)


def validate_episode(tensors):
    expected = {"chase_latents":(4,24,40),"ego_states":(3,)}
    u=tensors.get("uve_tokens")
    if u is None or u.ndim!=3 or u.shape[2]!=384 or u.shape[1]<=0 or u.shape[1]%6 or not torch.isfinite(u).all():raise ValueError("Invalid uve_tokens shape or values")
    n = len(u)
    for key,shape in expected.items():
        if key not in tensors or tuple(tensors[key].shape[1:]) != shape:
            raise ValueError(f"Invalid {key}: expected (frames,{shape})")
        tensor = tensors[key]
        if n is None:
            n = len(tensor)
        if len(tensor) != n or not torch.isfinite(tensor).all():
            raise ValueError("Inconsistent frame counts or non-finite cache data")
    if n < 17:
        raise ValueError("An episode needs at least 17 frames")
    if "timestamps" in tensors:
        ts=tensors["timestamps"]
        if ts.shape!=(n,) or not torch.isfinite(ts).all() or not (ts[1:]>ts[:-1]).all():
            raise ValueError("Invalid cache timestamps")
    if "camera_mask" in tensors:
        mask=tensors["camera_mask"]
        if mask.shape!=(n,6) or not mask.bool().any(-1).all():raise ValueError("Invalid camera mask")
        chunks=tensors["uve_tokens"].reshape(n,6,-1,384)
        if chunks[~mask.bool()].ne(0).any():raise ValueError("Unavailable cameras must have zero UVE tokens")
    return n


def load_episode(path):
    if Path(path).suffix != ".safetensors":
        raise ValueError("Only .safetensors episode caches are supported")
    tensors = load_file(str(path),device="cpu")
    validate_episode(tensors)
    meta=Path(path).with_name("metadata.json")
    episode_grid(tensors,json.loads(meta.read_text()) if meta.exists() else {})
    return tensors


def episode_grid(tensors,metadata):
    grid=metadata.get('uve_grid')
    count=tensors['uve_tokens'].shape[1]
    if grid is None:
        if count!=192:raise ValueError('Missing uve_grid for nonlegacy cache')
        grid=(4,8)
    grid=tuple(grid)
    if grid not in ((4,8),(5,8),(10,16)) or 6*grid[0]*grid[1]!=count:raise ValueError('UVE grid/token count mismatch')
    return grid


def require_uve_grid(config,dataset):
    if tuple(config.get('uve_grid',(4,8)))!=tuple(dataset.uve_grid):raise ValueError('Checkpoint/dataset UVE grid mismatch')
