"""Safe restart state, reproducible sample ordering, and EMA scheduling (no pickle)."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Sampler
from safetensors.torch import save_file,load_file
from .data.cache import atomic_json,resolve_safe_subpath


def ema_decay(target,step):
    if not 0<=target<1 or step<1:raise ValueError("Invalid EMA configuration")
    return min(target,(1+step)/(10+step))


def enable_training_backends(device):
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True


class RestartableSampler(Sampler):
    def __init__(self,count,seed=42,weights=None):
        self.count=count;self.weights=None if weights is None else torch.as_tensor(weights,dtype=torch.double)
        self.generator=torch.Generator().manual_seed(seed);self.order=[];self.cursor=0

    def __len__(self):return self.count

    def __iter__(self):
        if self.cursor>=len(self.order):
            self.order=(torch.randperm(self.count,generator=self.generator) if self.weights is None else
                        torch.multinomial(self.weights,self.count,replacement=True,generator=self.generator)).tolist()
            self.cursor=0
        while self.cursor<len(self.order):
            index=self.order[self.cursor];self.cursor+=1;yield index

    def state_dict(self):
        return dict(count=self.count,weights=self.weights,order=self.order,cursor=self.cursor,generator=self.generator.get_state())

    def load_state_dict(self,state):
        if state['count']!=self.count:raise ValueError("Resume dataset length changed")
        if (state['weights'] is None)!=(self.weights is None) or (self.weights is not None and not torch.equal(state['weights'],self.weights)):
            raise ValueError("Resume sampling configuration changed")
        self.order=state['order'];self.cursor=state['cursor'];self.generator.set_state(state['generator'])


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=np.random.get_state(),python=random.getstate(),
                mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])
    if state.get('mps') is not None and torch.backends.mps.is_available():torch.mps.set_rng_state(state['mps'])


def save_training_state(directory,state):
    """JSON tree + content-addressed tensor file; JSON is the atomic commit point."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);tensors={}
    def encode(value):
        if isinstance(value,(torch.Tensor,np.ndarray)):
            key=f'tensor_{len(tensors):06d}';array=isinstance(value,np.ndarray)
            tensor=torch.from_numpy(value.astype(np.int64)) if array and value.dtype==np.uint32 else torch.as_tensor(value)
            tensors[key]=tensor.detach().cpu().contiguous().clone()
            return dict(kind='numpy' if array else 'tensor',key=key,dtype=str(value.dtype))
        if isinstance(value,dict):return dict(kind='dict',items=[[encode(k),encode(v)] for k,v in value.items()])
        if isinstance(value,(list,tuple)):return dict(kind='tuple' if isinstance(value,tuple) else 'list',items=[encode(v) for v in value])
        if value is None or type(value) in (int,float,bool,str):return value
        raise TypeError(f'Unsupported restart value: {type(value)}')
    tree=encode(state);temporary=directory/'training_state.tmp'
    save_file(tensors,str(temporary));digest=hashlib.sha256(temporary.read_bytes()).hexdigest()
    filename=f'training_state_{digest}.safetensors';temporary.replace(directory/filename)
    atomic_json(directory/'training_state.json',dict(version=1,tensors=filename,sha256=digest,tree=tree))
    for stale in directory.glob('training_state_*.safetensors'):
        if stale.name!=filename:stale.unlink()


def load_training_state(directory):
    directory=Path(directory);metadata=json.loads((directory/'training_state.json').read_text())
    if metadata['version']!=1:raise ValueError('Unknown restart format')
    path=resolve_safe_subpath(directory,metadata['tensors'])
    if path.suffix!='.safetensors' or hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['sha256']:
        raise ValueError('Restart tensor fingerprint mismatch')
    tensors=load_file(str(path))
    def decode(value):
        if not isinstance(value,dict):return value
        kind=value['kind']
        if kind=='tensor':return tensors[value['key']]
        if kind=='numpy':return tensors[value['key']].numpy().astype(value['dtype'])
        if kind=='dict':return {decode(k):decode(v) for k,v in value['items']}
        if kind in ('list','tuple'):
            items=[decode(v) for v in value['items']];return tuple(items) if kind=='tuple' else items
        raise ValueError('Unknown restart node')
    return decode(metadata['tree'])


def cache_fingerprint(dataset):
    digest=hashlib.sha256()
    for path in dataset.paths:
        digest.update(str(path.relative_to(dataset.root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(path.with_name('metadata.json').read_bytes())
    return digest.hexdigest()
