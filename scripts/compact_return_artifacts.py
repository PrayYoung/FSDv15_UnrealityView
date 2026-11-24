"""Omit only an exact duplicate EMA export; keep the full restart state and best.

The latest EMA remains losslessly present in training_state tensors. This avoids
returning another ~327 MB copy while retaining exact resume, raw weights, EMA,
optimizer, scaler, RNG, sampler and the directly loadable best checkpoint.
"""
import argparse
from pathlib import Path
import torch
from safetensors import safe_open
from unrealityview.data.cache import atomic_json
from unrealityview.training import load_training_state


def compact(directory):
    directory=Path(directory);checkpoint=directory/'checkpoint';model=checkpoint/'model.safetensors'
    state=load_training_state(checkpoint)
    if not (directory/'checkpoint_best/model.safetensors').is_file():raise ValueError('Best checkpoint must be retained')
    if not model.exists():return
    expected={'dit.'+k:v for k,v in state['ema_model'].items()}
    expected.update({'policy.'+k:v for k,v in state['ema_policy'].items()})
    with safe_open(model,framework='pt',device='cpu') as tensors:
        if set(tensors.keys())!=set(expected):raise ValueError('Latest export does not match restart EMA keys')
        for name,value in expected.items():
            if not torch.equal(tensors.get_tensor(name),value.cpu()):raise ValueError('Latest export differs from restart EMA')
    size=model.stat().st_size;model.unlink()
    atomic_json(directory/'return_payload.json',dict(omitted_file='checkpoint/model.safetensors',omitted_bytes=size,
        reason='Exact duplicate of EMA in retained restart state',restart_complete=True,best_weights_retained=True,
        latest_export_command='python -m scripts.compact_return_artifacts --train TRAIN_DIRECTORY --export-latest'))


def export_latest(directory):
    from safetensors.torch import save_file
    directory=Path(directory);checkpoint=directory/'checkpoint';state=load_training_state(checkpoint)
    tensors={'dit.'+k:v.contiguous() for k,v in state['ema_model'].items()}
    tensors.update({'policy.'+k:v.contiguous() for k,v in state['ema_policy'].items()})
    target=checkpoint/'model.safetensors'
    if target.exists():raise ValueError('Latest export already exists')
    save_file(tensors,str(target),metadata={'step':str(state['step']),'format':'unrealityview-v1'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train',required=True);p.add_argument('--export-latest',action='store_true');a=p.parse_args()
    export_latest(a.train) if a.export_latest else compact(a.train)
