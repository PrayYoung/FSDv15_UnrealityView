"""Small trainable adapters; inference merges deltas into ordinary Linear layers.

Merging has a measurable one-off cost. The merged forward has no additional
LoRA branches; this is not a guarantee of identical wall-clock measurements.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from unrealityview.device import synchronize


class LoRALinear(nn.Module):
    def __init__(self,base:nn.Linear,rank=16,alpha=16.):
        super().__init__()
        if not isinstance(base,nn.Linear) or not 1<=rank<=min(base.in_features,base.out_features):
            raise ValueError("LoRA rank must fit a plain Linear layer")
        if not math.isfinite(alpha) or alpha<=0:raise ValueError("alpha must be finite and positive")
        self.base=base.requires_grad_(False)
        self.rank=int(rank)
        self.lora_a=nn.Parameter(base.weight.new_empty(rank,base.in_features))
        self.lora_b=nn.Parameter(base.weight.new_zeros(base.out_features,rank))
        self.register_buffer("scale",base.weight.new_tensor(float(alpha)/rank))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))

    @property
    def weight(self):return self.base.weight

    @property
    def bias(self):return self.base.bias

    def forward(self,x):
        result=self.base(x)
        delta=F.linear(F.linear(x,self.lora_a),self.lora_b)
        return result+delta.to(result.dtype)*self.scale.to(result.dtype)


def model_fingerprint(model):
    """Bind adapters to exact base parameters without storing a checkpoint path."""
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_targets(dit,start_block=4,include_unpatch=True):
    if not 0<=start_block<=dit.config.depth:raise ValueError("start_block is outside DiT depth")
    names=[]
    for index in range(start_block,dit.config.depth):
        names.extend(f"blocks.{index}.attn.{projection}" for projection in ("q","kv","out"))
        names.extend(f"blocks.{index}.mlp.{projection}" for projection in ("0","2"))
        if dit.config.cross_attention=="block":
            names.extend(f"blocks.{index}.cross_attn.{projection}" for projection in ("q","kv","out"))
    group_size=dit.config.depth//dit.config.stages
    # Cross-attention is stage-level in this architecture. Do not modify a
    # context group that also feeds an early protected geometric block.
    for stage in range(dit.config.stages):
        if dit.config.cross_attention=="stage" and stage*group_size>=start_block:
            names.extend(f"cross.{stage}.{projection}" for projection in ("q","kv","out"))
    if include_unpatch:names.append("output")
    if not names:raise ValueError("No adapter targets selected")
    return names


def inject_dit_lora(dit,rank=16,alpha=16.,start_block=4,include_unpatch=True,early_rank=None,early_blocks=4):
    if any(isinstance(m,LoRALinear) for m in dit.modules()):raise ValueError("DiT already contains LoRA layers")
    base_hash=model_fingerprint(dit)
    names=adapter_targets(dit,start_block,include_unpatch)
    modules=dict(dit.named_modules())
    if any(not isinstance(modules[name],nn.Linear) for name in names):raise ValueError("Expected plain Linear targets")
    # Validate all dimensions before mutating the model.
    if any(rank>min(modules[name].in_features,modules[name].out_features) for name in names):
        raise ValueError("Rank exceeds a target dimension")
    ranks={name:(early_rank if early_rank is not None and name.startswith("blocks.") and int(name.split(".")[1])<early_blocks else rank) for name in names}
    if any(type(r) is not int or not 1<=r<=min(modules[name].in_features,modules[name].out_features) for name,r in ranks.items()):
        raise ValueError("Invalid per-layer adapter rank")
    dit.requires_grad_(False)
    for name in names:
        parent,_,leaf=name.rpartition(".")
        setattr(dit.get_submodule(parent) if parent else dit,leaf,LoRALinear(modules[name],ranks[name],alpha))
    dit.lora_config=dict(rank=rank,alpha=float(alpha),start_block=start_block,include_unpatch=include_unpatch,
                         base_sha256=base_hash,targets=names,early_rank=early_rank,early_blocks=early_blocks)
    return dit


def extract_lora_state_dict(dit):
    result={}
    for name,module in dit.named_modules():
        if isinstance(module,LoRALinear):
            for key in ("lora_a","lora_b","scale"):
                result[f"{name}.{key}"]=getattr(module,key).detach().clone().contiguous()
    if not result:raise ValueError("No LoRA adapters found")
    return result


def save_lora(path,dit,style,step,state=None):
    if style not in ("comic","cyberpunk"):raise ValueError("Unsupported trained style")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state=extract_lora_state_dict(dit) if state is None else state
    metadata={"format":"unrealityview-lora-v1","style":style,"step":str(step),
              "config":json.dumps(dit.lora_config,sort_keys=True)}
    temporary=path.with_suffix(path.suffix+".tmp")
    save_file({k:v.detach().cpu().contiguous() for k,v in state.items()},str(temporary),metadata=metadata)
    temporary.replace(path)


@dataclass(frozen=True)
class Adapter:
    tensors:dict
    base_sha256:str
    fingerprint:str
    style:str
    size_bytes:int


def load_lora(path,expected_base=None):
    path=Path(path)
    if path.suffix!=".safetensors":raise ValueError("Adapters must use safetensors")
    with safe_open(path,framework="pt",device="cpu") as file:metadata=file.metadata() or {}
    if metadata.get("format")!="unrealityview-lora-v1":raise ValueError("Unknown adapter format")
    config=json.loads(metadata["config"])
    if expected_base is not None and config["base_sha256"]!=expected_base:raise ValueError("Adapter belongs to a different base model")
    tensors=load_file(str(path),device="cpu")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    return Adapter(tensors,config["base_sha256"],digest,metadata["style"],path.stat().st_size)


class LoRAWeightManager:
    """One owner must serialize merge + inference (the server uses its model lock)."""
    def __init__(self,base_dit):
        if any(isinstance(m,LoRALinear) for m in base_dit.modules()):raise ValueError("Inference manager requires a plain base DiT")
        self.model=base_dit
        self._originals={}
        self._signature=()
        self._base_hash=model_fingerprint(base_dit)
        base_dit.requires_grad_(False)

    @property
    def base_sha256(self):
        if self._base_hash is None:self._base_hash=model_fingerprint(self.model)
        return self._base_hash

    @torch.no_grad()
    def restore_base(self):
        for name,weight in self._originals.items():self.model.get_submodule(name).weight.copy_(weight)
        self._signature=()

    @staticmethod
    def _state_hash(state):
        digest=hashlib.sha256()
        for key,value in sorted(state.items()):
            digest.update(key.encode());digest.update(str(tuple(value.shape)).encode());digest.update(str(value.dtype).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def apply_adapters(self,adapters):
        """Merge [(name, Adapter or tensor dict, strength)], with no forward wrappers."""
        parsed=[];signature=[];size=0
        for name,adapter,strength in adapters:
            if not math.isfinite(strength) or not 0<=strength<=1.5:raise ValueError("Style strength must be in [0,1.5]")
            strength=round(float(strength),6)
            if isinstance(adapter,Adapter):
                if adapter.base_sha256!=self.base_sha256:raise ValueError("Adapter base fingerprint mismatch")
                state=adapter.tensors;fingerprint=adapter.fingerprint;size+=adapter.size_bytes
            else:
                state=adapter;fingerprint=self._state_hash(state);size+=sum(t.numel()*t.element_size() for t in state.values())
            signature.append((str(name),fingerprint,round(float(strength),6)))
            if strength:parsed.append((state,float(strength)))
        signature=tuple(signature)
        if signature==self._signature:return {"merge_ms":0.,"cache_hit":True,"adapter_mib":size/2**20}
        device=next(self.model.parameters()).device;synchronize(device);begin=time.perf_counter()
        deltas={}
        for state,strength in parsed:
            names={key.rsplit(".",1)[0] for key in state}
            for name in names:
                if set(k for k in state if k.startswith(name+"."))!={name+".lora_a",name+".lora_b",name+".scale"}:
                    raise ValueError("Adapter must contain only complete a/b/scale tensors")
                try:layer=self.model.get_submodule(name)
                except AttributeError as error:raise ValueError("Unknown adapter layer") from error
                if not isinstance(layer,nn.Linear):raise ValueError("Adapter target is not a Linear layer")
                a,b,scale=(state[name+"."+key] for key in ("lora_a","lora_b","scale"))
                if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=layer.in_features or b.shape!=(layer.out_features,a.shape[0]) or scale.numel()!=1:
                    raise ValueError("Adapter tensor shapes do not match the base")
                if not all(torch.isfinite(v).all() for v in (a,b,scale)):raise ValueError("Adapter has nonfinite tensors")
                delta=(b.float().to(device)@a.float().to(device))*(float(scale)*strength)
                deltas[name]=deltas.get(name,0)+delta
        # Build and validate candidates before changing any live inference weight.
        candidates={}
        for name in set(self._originals)|set(deltas):
            layer=self.model.get_submodule(name)
            if name not in self._originals:self._originals[name]=layer.weight.detach().clone()
            candidate=self._originals[name].float()+deltas.get(name,0)
            candidate=candidate.to(layer.weight.dtype)
            if not torch.isfinite(candidate).all():raise ValueError("Merged adapter overflows the base dtype")
            candidates[name]=candidate
        for name,weight in candidates.items():self.model.get_submodule(name).weight.copy_(weight)
        self._signature=signature;synchronize(device)
        return {"merge_ms":(time.perf_counter()-begin)*1000,"cache_hit":False,"adapter_mib":size/2**20}
