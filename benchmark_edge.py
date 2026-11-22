"""Measure 8-frame compute latency against the 222.2ms budget; no FPS guarantees."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from fsd_reality.checkpoint import load_checkpoint
from fsd_reality.data.cache import atomic_json
from fsd_reality.device import select_device,synchronize,autocast_for
from fsd_reality.models.meanflow_dit import ConditionalMeanFlowDiT
from fsd_reality.models.driving_policy_head import DrivingPolicyHead
from fsd_reality.models.pretrained_taesd import PretrainedTAESD


def measure(fn,device,repeats=5,warmup=2):
    for _ in range(warmup):fn()
    synchronize(device)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    samples=[]
    for _ in range(repeats):
        synchronize(device);start=time.perf_counter();fn();synchronize(device)
        samples.append((time.perf_counter()-start)*1000)
    return {"median_ms":statistics.median(samples),"min_ms":min(samples),"samples_ms":samples,
            "peak_vram_mib":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else None}


@torch.no_grad()
def benchmark(args):
    device=select_device(args.device)
    if args.checkpoint:
        model,policy,config=load_checkpoint(args.checkpoint,device)
        weights=f"EMA step {config['step']}"
    else:
        model,policy=ConditionalMeanFlowDiT().to(device).eval(),DrivingPolicyHead().to(device).eval()
        weights="random initialization: timings only"
    decoder=PretrainedTAESD().to(device)
    shape=(1,4,*model.config.latent_shape)
    previous=torch.randn(shape,device=device)
    tokens=torch.randn(1,4,192,384,device=device)
    noise=torch.randn_like(previous)
    ages=torch.full((1,4),1/36,device=device)
    report={"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
            "torch":torch.__version__,"weights":weights,"budget_ms":8/36*1000,"batch":1,
            "dit_config":model.config.to_dict(),"sram":decoder.sram_budget_report(),"variants":{}}
    with autocast_for(device):
        policy_result=measure(lambda:policy(tokens[:,0]),device,args.repeats)
    report["decoupled_policy"]=policy_result
    for nfe in (1,2):
        for tiled in (False,True):
            decode=decoder.decode_tiled if tiled else decoder.decode
            def generate():
                with autocast_for(device):latent=model.sample_chunk(previous,tokens,nfe,ages,noise=noise)
                return decode(latent.float())
            result=measure(generate,device,args.repeats)
            result["policy_plus_video_ms"]=result["median_ms"]+policy_result["median_ms"]
            result["headroom_ms"]=report["budget_ms"]-result["policy_plus_video_ms"]
            result["effective_fps"]=8000/result["policy_plus_video_ms"]
            report["variants"][f"{nfe}_nfe_{'tiled' if tiled else 'full'}"]=result
    full,tiled=decoder.decode(previous),decoder.decode_tiled(previous)
    report["tiled_equivalence"]={"max_abs_error":(full-tiled).abs().max().item(),
                                    "mean_abs_error":(full-tiled).abs().mean().item()}
    atomic_json(args.output,report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--device",default="auto")
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--output",default="outputs/benchmark.json")
    a=p.parse_args()
    if a.repeats<1:p.error("repeats must be positive")
    benchmark(a)


if __name__=="__main__":main()
