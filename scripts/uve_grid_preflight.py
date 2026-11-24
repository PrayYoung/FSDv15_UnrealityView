"""Twenty optimizer steps at full JVP strength; separate process from real training."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from unrealityview.data.dataset import CachedEpisodeChunkDataset
from unrealityview.data.cache import atomic_json
from unrealityview.device import autocast_for
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.objectives import meanflow_loss
from unrealityview.training import enable_training_backends,clip_training_gradients


def run(data,config,batch,accum):
    device=torch.device('cuda');random.seed(42);np.random.seed(42);torch.manual_seed(42)
    enable_training_backends(device)
    dataset=CachedEpisodeChunkDataset(data,'train',jitter=True)
    loader=iter(DataLoader(dataset,batch_size=batch,shuffle=True,num_workers=0))
    model=ConditionalMeanFlowDiT(DiTConfig(**config['dit'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.01)
    # Peak memory includes native exact-JVP derivative at full training strength,
    # rather than only the first 20 low-weight warmup updates.
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            item=next(loader);b={k:item[k].to(device) for k in ('z0','z_prev','uve_tokens','stage_ages')}
            with autocast_for(device):
                loss,_=meanflow_loss(model,b['z0'],b['z_prev'],b['uve_tokens'],b['stage_ages'],jvp_method=config['jvp_method'],
                    fm_ratio=config['fm_ratio'],adaptive_p=config['adaptive_p'],adaptive_c=config['adaptive_c'],time_sampling=config['time_sampling'],jvp_weight=1.)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite preflight loss')
            (loss/accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    torch.cuda.synchronize();seconds=time.monotonic()-start
    extra=sum(p.numel()*p.element_size() for p in model.parameters())+64*2**20
    total=torch.cuda.get_device_properties(0).total_memory
    safe=torch.cuda.max_memory_allocated()+extra<=.94*total
    return dict(status='passed' if safe else 'insufficient_headroom',ema_policy_decoder_reserve_bytes=extra,steps=20,batch_size=batch,grad_accum=accum,effective_batch=batch*accum,
        uve_grid=list(dataset.uve_grid),seconds=seconds,steps_per_second=20/seconds,
        projected_training_seconds=seconds/20*5000,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        note='Full-JVP stress check; no checkpoint or RNG state reused for real training. Projection excludes evaluation/checkpoint IO.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',required=True);p.add_argument('--batch-size',type=int,required=True);p.add_argument('--grad-accum',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:result=run(a.data,json.loads(Path(a.config).read_text()),a.batch_size,a.grad_accum)
    except torch.cuda.OutOfMemoryError:
        atomic_json(a.output,dict(status='out_of_memory',batch_size=a.batch_size,grad_accum=a.grad_accum));raise SystemExit(42)
    atomic_json(a.output,result)
    if result["status"]!="passed":raise SystemExit(42)


if __name__=='__main__':main()
