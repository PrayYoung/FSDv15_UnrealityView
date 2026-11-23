"""Measure actual style merge cost and merged forward timings; no zero-ms claims."""
import argparse
from pathlib import Path
import torch
from torch import nn
from benchmark_edge import measure
from unrealityview.checkpoint import load_checkpoint
from unrealityview.data.cache import atomic_json
from unrealityview.device import select_device,autocast_for
from unrealityview.models.lora import LoRAWeightManager,load_lora,model_fingerprint


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",default="outputs/real_train/checkpoint")
    p.add_argument("--lora-dir",default="outputs/adapters");p.add_argument("--output",default="outputs/style_benchmark.json")
    p.add_argument("--device",default="auto");p.add_argument("--repeats",type=int,default=7);a=p.parse_args()
    if a.repeats<3:p.error("Use at least three timed repeats")
    device=select_device(a.device);model,policy,config=load_checkpoint(a.base,device)
    manager=LoRAWeightManager(model);fingerprint=model_fingerprint(policy)
    adapters={}
    for style in ("comic","cyberpunk"):
        candidates=[Path(a.lora_dir)/f"{style}_lora.safetensors",Path(a.lora_dir)/style/f"{style}_lora.safetensors"]
        path=next((p for p in candidates if p.is_file()),None)
        if path is None:raise ValueError(f"Missing {style} adapter")
        adapters[style]=load_lora(path,manager.base_sha256)
    generator=torch.Generator(device=device).manual_seed(52)
    previous=torch.randn((1,4,*model.config.latent_shape),device=device,generator=generator)
    context=torch.randn((1,4,192,384),device=device,generator=generator);noise=torch.randn(previous.shape,device=device,generator=generator)
    result={"base_step":config["step"],"device":str(device),"samples":a.repeats,"styles":{},
            "forward_structure":"plain Linear layers; no LoRA forward branches",
            "timing_note":"Merged DiT compute only; merge cost reported separately; timings may vary"}
    layers=sum(isinstance(m,nn.Linear) for m in model.modules())
    for style in ("reality","comic","cyberpunk","hybrid"):
        requested=[] if style=="reality" else [(name,adapters[name],.55 if style=="hybrid" else 1.)
                      for name in (("comic","cyberpunk") if style=="hybrid" else (style,))]
        merge=manager.apply_adapters(requested)
        with autocast_for(device):timing=measure(lambda:model.sample_chunk(previous,context,1,noise=noise),device,a.repeats,2)
        result["styles"][style]={**timing,**merge}
        assert sum(isinstance(m,nn.Linear) for m in model.modules())==layers
    manager.restore_base();result["policy_unchanged"]=model_fingerprint(policy)==fingerprint
    atomic_json(a.output,result)


if __name__=="__main__":main()
