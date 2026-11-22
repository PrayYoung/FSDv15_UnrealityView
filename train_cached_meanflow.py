"""Train cached Mean-Flow DiT and independent policy together; save safe EMA weights."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from fsd_reality.checkpoint import save_checkpoint
from fsd_reality.data.cache import atomic_json
from fsd_reality.data.dataset import CachedEpisodeChunkDataset
from fsd_reality.device import select_device,precision_for,autocast_for
from fsd_reality.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from fsd_reality.models.driving_policy_head import DrivingPolicyHead,training_goals
from fsd_reality.models.pretrained_taesd import PretrainedTAESD
from fsd_reality.objectives import meanflow_loss
from evaluate import evaluate_models


def train(args):
    config=json.loads(Path(args.config).read_text())
    if args.steps is not None: config["steps"]=args.steps
    if args.batch_size is not None: config["batch_size"]=args.batch_size
    device=select_device(args.device)
    seed=config["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/"run_config.json",{**config,"device":str(device),"precision":str(precision_for(device))})
    train_data=CachedEpisodeChunkDataset(args.data,"train")
    validation=CachedEpisodeChunkDataset(args.data,"val",jitter=False)
    manifest=Path(args.data)/"manifest.json"
    data_source=json.loads(manifest.read_text()).get("source","unknown") if manifest.is_file() else "unknown"
    loader=DataLoader(train_data,batch_size=config["batch_size"],shuffle=True,num_workers=0)
    model=ConditionalMeanFlowDiT(DiTConfig(**config["dit"])).to(device)
    policy=DrivingPolicyHead().to(device)
    ema_model,ema_policy=copy.deepcopy(model).eval().requires_grad_(False),copy.deepcopy(policy).eval().requires_grad_(False)
    parameters=list(model.parameters())+list(policy.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=config["learning_rate"],weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and precision_for(device)==torch.float16)
    decoder=PretrainedTAESD().to(device)
    iterator=iter(loader)
    start=time.monotonic()
    metrics_file=output/"train_metrics.jsonl"
    for step in range(1,config["steps"]+1):
        model.train();policy.train();optimizer.zero_grad(set_to_none=True)
        warmup=config["warmup_steps"]
        progress=max(0,(step-warmup)/max(1,config["steps"]-warmup))
        factor=min(1,step/max(1,warmup))*.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups: group["lr"]=config["learning_rate"]*factor
        totals={"loss":0.,"flow_loss":0.,"policy_loss":0.}
        for _ in range(config["grad_accum"]):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader);batch=next(iterator)
            b={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            with autocast_for(device):
                flow,details=meanflow_loss(model,b["z0"],b["z_prev"],b["uve_tokens"],b["stage_ages"],jvp_method=config["jvp_method"])
                goal=training_goals(b["waypoints"])
                predicted=policy(b["uve_tokens"][:,0],goal)
                policy_loss=torch.nn.functional.smooth_l1_loss(predicted,b["waypoints"])
                loss=flow+config["policy_weight"]*policy_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at optimizer step {step}")
            scaler.scale(loss/config["grad_accum"]).backward()
            for key,value in (("loss",loss),("flow_loss",flow),("policy_loss",policy_loss)):
                totals[key]+=value.detach().item()/config["grad_accum"]
        scaler.unscale_(optimizer)
        grad_norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Nonfinite gradient at step {step}")
        scaler.step(optimizer);scaler.update()
        with torch.no_grad():
            for ema,current in ((ema_model,model),(ema_policy,policy)):
                for old,new in zip(ema.parameters(),current.parameters()):
                    old.lerp_(new,1-config["ema_decay"])
        row={"step":step,**totals,"grad_norm":grad_norm.item(),"lr":optimizer.param_groups[0]["lr"],
             "elapsed_sec":time.monotonic()-start}
        with metrics_file.open("a") as f: f.write(json.dumps(row,allow_nan=False)+"\n")
        if step==1 or step%25==0: print(json.dumps(row),flush=True)
        if step%config["eval_every"]==0 or step==config["steps"]:
            save_checkpoint(output/"checkpoint",ema_model,ema_policy,step,
                            {"training_config":config,"weights":"EMA","data_source":data_source})
            report=evaluate_models(ema_model,ema_policy,decoder,validation,device,
                                   output/f"eval_{step:06d}",config["eval_items"])
            print("EVALUATION",json.dumps(report["nfe"]),flush=True)
    atomic_json(output/"summary.json",{"status":"completed","steps":config["steps"],
                "seconds":time.monotonic()-start,"last_training_metrics":row,
                "parameters_dit":sum(p.numel() for p in model.parameters()),
                "parameters_policy":sum(p.numel() for p in policy.parameters())})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/train.json")
    p.add_argument("--data",default="outputs/dataset")
    p.add_argument("--output",default="outputs/train")
    p.add_argument("--device",default="auto")
    p.add_argument("--steps",type=int)
    p.add_argument("--batch-size",type=int)
    a=p.parse_args()
    if a.steps is not None and a.steps<1: p.error("steps must be positive")
    if a.batch_size is not None and a.batch_size<1: p.error("batch size must be positive")
    train(a)


if __name__=="__main__":
    main()
