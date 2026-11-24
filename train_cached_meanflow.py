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
from unrealityview.checkpoint import save_checkpoint
from unrealityview.data.cache import atomic_json
from unrealityview.data.dataset import CachedEpisodeChunkDataset
from unrealityview.device import select_device,precision_for,autocast_for
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.driving_policy_head import DrivingPolicyHead,training_goals
from unrealityview.models.pretrained_taesd import PretrainedTAESD
from unrealityview.objectives import meanflow_loss
from evaluate import evaluate_models
from unrealityview.training import (ema_decay,enable_training_backends,RestartableSampler,
    rng_state,restore_rng,save_training_state,load_training_state,cache_fingerprint,
    clip_training_gradients,initialize_weights)


def train(args):
    config=json.loads(Path(args.config).read_text())
    if args.steps is not None: config["steps"]=args.steps
    if args.batch_size is not None: config["batch_size"]=args.batch_size
    if getattr(args,"grad_accum",None) is not None:config["grad_accum"]=args.grad_accum
    device=select_device(args.device)
    enable_training_backends(device)
    seed=config["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/"run_config.json",{**config,"device":str(device),"precision":str(precision_for(device))})
    train_data=CachedEpisodeChunkDataset(args.data,"train",jitter=config.get("train_jitter",True))
    windows=config.get("diagnostic_train_windows")
    if windows is not None:
        if type(windows) is not int or windows<1:raise ValueError("diagnostic_train_windows must be positive")
        train_data.index=train_data.index[:windows]*max(1,math.ceil(config["batch_size"]/min(windows,len(train_data.index))))
    validation=CachedEpisodeChunkDataset(args.data,"val",jitter=False)
    if train_data.uve_grid!=validation.uve_grid:raise ValueError("Train/validation UVE grid mismatch")
    manifest=Path(args.data)/"manifest.json"
    data_source=json.loads(manifest.read_text()).get("source","unknown") if manifest.is_file() else "unknown"
    weights=None
    if config.get("balance_sources",False):
        from collections import Counter
        sources=[train_data.metadata[ep].get("source","unknown") for ep,_ in train_data.index]
        counts=Counter(sources)
        weights=[1/counts[s] for s in sources]
    sampler=RestartableSampler(len(train_data),seed,weights)
    loader=DataLoader(train_data,batch_size=config["batch_size"],sampler=sampler,num_workers=0,
                      pin_memory=device.type=="cuda",generator=torch.Generator().manual_seed(seed))
    provenance={"train_source_ids":sorted(k for k,v in train_data.provenance.items() if v=="train"),
                "validation_source_ids":sorted(k for k,v in train_data.provenance.items() if v=="val")}
    dit_config=dict(config["dit"])
    # Archived configurations retain the historical embedding semantics.
    dit_config.setdefault("time_scale",1000.)
    model=ConditionalMeanFlowDiT(DiTConfig(**dit_config)).to(device)
    model.uve_grid=train_data.uve_grid
    policy=DrivingPolicyHead().to(device)
    init=getattr(args,"init_weights",None)
    if init and getattr(args,"resume",None):raise ValueError("init-weights and resume are mutually exclusive")
    if init:atomic_json(output/"initialization.json",initialize_weights(model,policy,init))
    ema_model,ema_policy=copy.deepcopy(model).eval().requires_grad_(False),copy.deepcopy(policy).eval().requires_grad_(False)
    parameters=list(model.parameters())+list(policy.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=config["learning_rate"],weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and precision_for(device)==torch.float16)
    decoder=PretrainedTAESD().to(device)
    start_step=0;best_psnr=-float("inf")
    fingerprints={"train":cache_fingerprint(train_data),"val":cache_fingerprint(validation)}
    resume=getattr(args,"resume",None)
    if resume:
        state=load_training_state(resume)
        if state["config"]!=config or state["data_fingerprints"]!=fingerprints:
            raise ValueError("Resume configuration or cached data changed; use the original training schedule")
        model.load_state_dict(state["model"]);policy.load_state_dict(state["policy"])
        ema_model.load_state_dict(state["ema_model"]);ema_policy.load_state_dict(state["ema_policy"])
        optimizer.load_state_dict(state["optimizer"]);scaler.load_state_dict(state["scaler"])
        sampler.load_state_dict(state["sampler"]);restore_rng(state["rng"])
        start_step=state["step"];best_psnr=state["best_psnr"]
    if start_step>=config["steps"]:raise ValueError("Resume checkpoint already reached the configured total steps")
    iterator=iter(loader)
    if getattr(args,"evaluate_initial",False) and not resume:
        evaluate_models(ema_model,ema_policy,decoder,validation,device,output/"eval_000000",config["eval_items"])
    if config.get("diagnostic_train_eval",False) and not resume:
        evaluate_models(ema_model,ema_policy,decoder,train_data,device,output/"overfit_000000",1)
    start=time.monotonic()
    metrics_file=output/"train_metrics.jsonl"
    stop_at=min(config["steps"],getattr(args,"stop_after",None) or config["steps"])
    if stop_at<=start_step:raise ValueError("stop-after must exceed the resume step")
    for step in range(start_step+1,stop_at+1):
        model.train();policy.train();optimizer.zero_grad(set_to_none=True)
        warmup=config["warmup_steps"]
        progress=max(0,(step-warmup)/max(1,config["steps"]-warmup))
        factor=min(1,step/max(1,warmup))*.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups: group["lr"]=config["learning_rate"]*factor
        totals={"loss":0.,"flow_loss":0.,"policy_loss":0.}
        diagnostics={};backends=set()
        warmup_jvp=config.get("jvp_warmup_steps",0)
        jvp_weight=min(1.,max(0.,(step-1)/warmup_jvp)) if warmup_jvp else 1.
        legacy="time_scale" not in config["dit"]
        options=dict(fm_ratio=config.get("fm_ratio",.5 if legacy else .75),
            adaptive_p=config.get("adaptive_p",.5 if legacy else 1.),
            adaptive_c=config.get("adaptive_c",.001 if legacy else .01),
            time_sampling=config.get("time_sampling","uniform" if legacy else "logit_normal"),
            jvp_weight=jvp_weight)
        for _ in range(config["grad_accum"]):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader);batch=next(iterator)
            model_inputs={"z0","z_prev","uve_tokens","stage_ages","waypoints","policy_tokens"}
            b={k:v.to(device,non_blocking=device.type=="cuda") if k in model_inputs and isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            with autocast_for(device):
                flow,details=meanflow_loss(model,b["z0"],b["z_prev"],b["uve_tokens"],b["stage_ages"],jvp_method=config["jvp_method"],**options)
                if config.get("policy_weight",1.) != 0:
                    goal=training_goals(b["waypoints"])
                    predicted=policy(b["policy_tokens"],goal)
                    policy_loss=torch.nn.functional.smooth_l1_loss(predicted,b["waypoints"])
                else:policy_loss=flow.new_zeros(())
                loss=flow+config["policy_weight"]*policy_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at optimizer step {step}")
            backends.add(details["jvp_backend"])
            for key,value in details.items():
                if key!="jvp_backend":diagnostics[key]=diagnostics.get(key,0.)+float(value)/config["grad_accum"]
            scaler.scale(loss/config["grad_accum"]).backward()
            for key,value in (("loss",loss),("flow_loss",flow),("policy_loss",policy_loss)):
                totals[key]+=value.detach().item()/config["grad_accum"]
        scaler.unscale_(optimizer)
        gradient_stats=clip_training_gradients(model,policy)
        scaler.step(optimizer);scaler.update()
        with torch.no_grad():
            for ema,current in ((ema_model,model),(ema_policy,policy)):
                for old,new in zip(ema.parameters(),current.parameters()):
                    old.lerp_(new,1-ema_decay(config["ema_decay"],step))
        row={"step":step,**totals,**diagnostics,**gradient_stats,"jvp_backend":"+".join(sorted(backends)),
             "precision":str(precision_for(device)),"lr":optimizer.param_groups[0]["lr"],
             "elapsed_sec":time.monotonic()-start}
        with metrics_file.open("a") as f: f.write(json.dumps(row,allow_nan=False)+"\n")
        if step==1 or step%25==0: print(json.dumps(row),flush=True)
        if step%config["eval_every"]==0 or step==stop_at:
            save_checkpoint(output/"checkpoint",ema_model,ema_policy,step,
                            {"training_config":config,"weights":"EMA","data_source":data_source,
                             "data_provenance":provenance,
                             "target_views":sorted(set(m.get("target_view","elevated_chase") for m in train_data.metadata))})
            report=evaluate_models(ema_model,ema_policy,decoder,validation,device,
                                   output/f"eval_{step:06d}",config["eval_items"])
            score=report["nfe"]["1"]["psnr_db"]
            if score>best_psnr:
                best_psnr=score
                save_checkpoint(output/"checkpoint_best",ema_model,ema_policy,step,
                    {"training_config":config,"weights":"EMA","data_source":data_source,"data_provenance":provenance,
                     "validation_psnr_db":score,
                     "target_views":sorted(set(m.get("target_view","elevated_chase") for m in train_data.metadata))})
            save_training_state(output/"checkpoint",dict(step=step,config=config,data_fingerprints=fingerprints,
                model=model.state_dict(),policy=policy.state_dict(),ema_model=ema_model.state_dict(),
                ema_policy=ema_policy.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                sampler=sampler.state_dict(),rng=rng_state(),best_psnr=best_psnr))
            if config.get("diagnostic_train_eval",False):
                evaluate_models(ema_model,ema_policy,decoder,train_data,device,output/f"overfit_{step:06d}",1)
            if config.get("eval_raw_comparison",False):
                evaluate_models(model,policy,decoder,validation,device,output/f"eval_raw_{step:06d}",1)
            print("EVALUATION",json.dumps(report["nfe"]),flush=True)
    atomic_json(output/"summary.json",{"status":"completed" if step==config["steps"] else "stopped_at_checkpoint","steps":step,
                "planned_steps":config["steps"],
                "seconds":time.monotonic()-start,"last_training_metrics":row,
                "parameters_dit":sum(p.numel() for p in model.parameters()),
                "parameters_policy":sum(p.numel() for p in policy.parameters())})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/real_train.json")
    p.add_argument("--data",default="outputs/real_dataset")
    p.add_argument("--output",default="outputs/real_train")
    p.add_argument("--device",default="auto")
    p.add_argument("--stop-after",type=int,help="Stop safely at this optimizer step without changing the planned schedule")
    p.add_argument("--evaluate-initial",action="store_true",help="Record a fixed-noise validation baseline before training")
    p.add_argument("--init-weights",help="Weight-only warm start with fresh optimizer and step counter")
    p.add_argument("--resume",help="Safe training-state checkpoint directory; retain the original total-step schedule")
    p.add_argument("--steps",type=int)
    p.add_argument("--batch-size",type=int)
    p.add_argument("--grad-accum",type=int)
    a=p.parse_args()
    if a.steps is not None and a.steps<1: p.error("steps must be positive")
    if a.batch_size is not None and a.batch_size<1: p.error("batch size must be positive")
    if a.grad_accum is not None and a.grad_accum<1:p.error("grad accumulation must be positive")
    train(a)


if __name__=="__main__":
    main()
