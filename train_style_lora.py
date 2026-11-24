"""Train only video-style adapters. A trained base is needed for meaningful output."""
from unrealityview.styles import validate_style_slug
import argparse
import json
import math
import random
import hashlib
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.utils.data import DataLoader
from unrealityview.checkpoint import load_checkpoint
from unrealityview.data.cache import atomic_json
from unrealityview.data.dataset import stage_context
from unrealityview.data.stylize_cache import StylizedChunkDataset
from unrealityview.device import select_device,autocast_for,precision_for
from unrealityview.models.lora import inject_dit_lora,extract_lora_state_dict,save_lora,model_fingerprint
from unrealityview.models.pretrained_taesd import PretrainedTAESD
from unrealityview.objectives import meanflow_loss
from unrealityview.visuals import to_image,save_rollout
from evaluate import pixel_metrics
from unrealityview.training import (ema_decay,enable_training_backends,RestartableSampler,rng_state,
    restore_rng,save_training_state,load_training_state,cache_fingerprint)


def save_style_sheet(rows,path,frames=(0,3,7)):
    w,h=rows[0][1].shape[-1],rows[0][1].shape[-2]
    image=Image.new("RGB",(w*len(frames),(h+26)*len(rows)),"#111820");draw=ImageDraw.Draw(image)
    for row,(label,video) in enumerate(rows):
        for col,index in enumerate(frames):
            x,y=col*w,row*(h+26);draw.text((x+6,y+6),label,fill="white");image.paste(to_image(video[:,index]),(x,y+26))
    image.save(path)


@torch.no_grad()
def evaluate_style(model,policy,decoder,data,device,output,max_items=4,seed=719):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);model.eval();rows=[]
    for index in range(min(max_items,len(data))):
        item=data[index];real=data.base[index]
        tokens=item["uve_tokens"][None].to(device);ages=item["stage_ages"][None].to(device)
        styled_prev=item["z_prev"][None].to(device);real_prev=real["z_prev"][None].to(device)
        episode,start=data.base.index[index]
        images=[np.array(Image.open(data.directories[episode]/"frames"/f"{start+i:05d}.png").convert("RGB")) for i in range(8)]
        truth=torch.from_numpy(np.stack(images)).permute(3,0,1,2)[None].float().to(device)/127.5-1
        generator=torch.Generator(device=device).manual_seed(seed+index)
        noise=torch.randn(styled_prev.shape,device=device,generator=generator)
        with autocast_for(device):
            steady=model.sample_chunk(styled_prev,tokens,1,ages,noise=noise)
            switch=model.sample_chunk(real_prev,tokens,1,ages,noise=noise)
            waypoints=policy(item["policy_tokens"][None].to(device))
        steady_pixels=decoder.decode(steady.float());switch_pixels=decoder.decode(switch.float())
        rows.append({"steady":pixel_metrics(steady_pixels,truth),"switch":pixel_metrics(switch_pixels,truth),
                     "waypoint_l1_m":(waypoints[0,:,:2]-item["waypoints"][:,:2].to(device)).abs().mean().item(),
                     "source_id":item["source_id"],"start":start})
        if index==0:
            save_style_sheet([("Previous real",decoder.decode(real_prev)[0]),("Style teacher",truth[0]),
                              ("1-NFE steady",steady_pixels[0]),("1-NFE real to style",switch_pixels[0])],output/"contact_sheet.png")
    report={"split":"val","samples":len(rows),"seed":seed,"per_chunk":rows,
            "reference":"optical-flow stabilized styled PNGs from real PandaSet",
            "note":"Adapter validation, not a physical driving-control or zero-flicker guarantee"}
    atomic_json(output/"metrics.json",report)
    episode=data.base.episodes[0];previous=episode["chase_latents"][:8].permute(1,0,2,3)[None].float().to(device)
    frames=[];generator=torch.Generator(device=device).manual_seed(seed+10000)
    for chunk in range(4):
        start=8+chunk*8;ids,ages=stage_context(episode,start,float(data.base.metadata[0].get("fps",10)))
        tokens=episode["uve_tokens"][ids][None].float().to(device)
        with autocast_for(device):
            previous=model.sample_chunk(previous,tokens,1,ages[None].to(device),
                         noise=torch.randn(previous.shape,device=device,generator=generator))
        frames.append(decoder.decode(previous.float())[0]);previous=previous.float()
    rollout=torch.cat(frames,1)
    save_rollout(rollout,output/"closed_loop.gif",fps=float(data.base.metadata[0].get("fps",10)))
    save_style_sheet([("1-NFE real to style closed loop",rollout)],output/"rollout.png",(0,7,15,23,31))
    return report


def train(args):
    device=select_device(args.device);enable_training_backends(device)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    model,policy,base_config=load_checkpoint(args.base,device)
    policy.eval().requires_grad_(False);policy_before=model_fingerprint(policy)
    inject_dit_lora(model,args.rank,args.alpha,args.start_block,not args.no_unpatch,getattr(args,"early_rank",None))
    params=[p for p in model.parameters() if p.requires_grad]
    ema=extract_lora_state_dict(model)
    data=StylizedChunkDataset(args.data,args.styles_data,args.style,"train",.2)
    validation=StylizedChunkDataset(args.data,args.styles_data,args.style,"val",0.)
    train_ids=sorted(k for k,v in data.base.provenance.items() if v=="train")
    # Adapters cannot silently train on scenes reserved as base validation/test.
    forbidden=set(base_config.get("data_provenance",{}).get("validation_source_ids",[]))
    if forbidden&set(train_ids):raise ValueError("Style train scenes overlap base validation scenes")
    sampler=RestartableSampler(len(data),args.seed)
    loader=DataLoader(data,batch_size=args.batch_size,sampler=sampler,num_workers=0,
                      pin_memory=device.type=="cuda",generator=torch.Generator().manual_seed(args.seed))
    optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and precision_for(device)==torch.float16)
    decoder=PretrainedTAESD().to(device)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    config=dict(style=args.style,steps=args.steps,rank=args.rank,alpha=args.alpha,start_block=args.start_block,
                base_step=base_config["step"],base_sha256=model.lora_config["base_sha256"],train_source_ids=train_ids,
                adapter_parameters=sum(p.numel() for p in params),effective_batch=args.batch_size*args.grad_accum,
                previous_real_ratio=.2,previous_style_ratio=.8,previous_noise_max=.25,objective="improved Mean-Flow",
                base_quality_note="A short/smoke base does not qualify a style demo",policy_sha256=policy_before)
    config["hyperparameters"]={k:v for k,v in vars(args).items() if k not in
                                  ("base","data","styles_data","output","resume","device","stop_after")}
    fingerprints={split:{"base":cache_fingerprint(ds.base),"styles":[hashlib.sha256((d/"style.safetensors").read_bytes()).hexdigest() for d in ds.directories]}
                  for split,ds in (("train",data),("val",validation))}
    start_step=0;best_psnr=-float("inf")
    if getattr(args,"resume",None):
        state=load_training_state(args.resume)
        if state["config"]!=config or state["data_fingerprints"]!=fingerprints:raise ValueError("Resume configuration or style data changed")
        named=dict(model.named_parameters())
        with torch.no_grad():
            for key,value in state["adapter"].items():
                if key in named:named[key].copy_(value)
        ema=state["ema"];ema={k:v.to(device) for k,v in ema.items()}
        optimizer.load_state_dict(state["optimizer"]);scaler.load_state_dict(state["scaler"])
        sampler.load_state_dict(state["sampler"]);restore_rng(state["rng"])
        start_step=state["step"];best_psnr=state["best_psnr"]
    if start_step>=args.steps:raise ValueError("Resume checkpoint already reached the configured total steps")
    iterator=iter(loader)
    atomic_json(output/"config.json",config);start=time.monotonic()
    stop_at=min(args.steps,getattr(args,"stop_after",None) or args.steps)
    if stop_at<=start_step:raise ValueError("stop-after must exceed the resume step")
    for step in range(start_step+1,stop_at+1):
        model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.
        fraction=max(0.,(step-args.warmup)/max(1,args.steps-args.warmup))
        lr=args.lr*min(1,step/max(1,args.warmup))*.5*(1+math.cos(math.pi*fraction))
        for group in optimizer.param_groups:group["lr"]=lr
        for _ in range(args.grad_accum):
            try:item=next(iterator)
            except StopIteration:iterator=iter(loader);item=next(iterator)
            fields={k:item[k].to(device,non_blocking=device.type=="cuda") for k in ("z0","z_prev","uve_tokens","stage_ages")}
            with autocast_for(device):
                loss,_=meanflow_loss(model,fields["z0"],fields["z_prev"],fields["uve_tokens"],fields["stage_ages"],
                                    prev_chunk_noise_std_max=.25,jvp_method="auto")
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite adapter loss")
            scaler.scale(loss/args.grad_accum).backward();loss_sum+=float(loss.detach())/args.grad_accum
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise RuntimeError("Nonfinite adapter gradients")
        scaler.step(optimizer);scaler.update()
        with torch.no_grad():
            current=extract_lora_state_dict(model)
            for key in ema:ema[key].lerp_(current[key],1-ema_decay(args.ema,step)) if not key.endswith(".scale") else ema[key].copy_(current[key])
        row=dict(step=step,loss=loss_sum,grad_norm=float(norm),lr=lr,elapsed_sec=time.monotonic()-start)
        with (output/"metrics.jsonl").open("a") as file:file.write(json.dumps(row,allow_nan=False)+"\n")
        if step==1 or step%25==0:print(json.dumps(row),flush=True)
        if step%args.eval_every==0 or step==stop_at:
            save_lora(output/(args.style+"_lora.safetensors"),model,args.style,step,ema)
            current=extract_lora_state_dict(model)
            with torch.no_grad():
                named=dict(model.named_parameters())
                for key,value in ema.items():
                    if key in named:named[key].copy_(value)
            report=evaluate_style(model,policy,decoder,validation,device,output/f"eval_{step:06d}",args.eval_items,args.seed)
            score=float(np.mean([row["steady"]["psnr_db"] for row in report["per_chunk"]]))
            if score>best_psnr:
                best_psnr=score
                save_lora(output/"checkpoint_best"/(args.style+"_lora.safetensors"),model,args.style,step,ema)
                atomic_json(output/"checkpoint_best"/"metrics.json",dict(step=step,validation_psnr_db=score))
            with torch.no_grad():
                for key,value in current.items():
                    if key in named:named[key].copy_(value)
            save_lora(output/"checkpoint"/(args.style+"_lora.safetensors"),model,args.style,step,ema)
            save_training_state(output/"checkpoint",dict(step=step,config=config,data_fingerprints=fingerprints,
                adapter=extract_lora_state_dict(model),ema=ema,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                sampler=sampler.state_dict(),rng=rng_state(),best_psnr=best_psnr))
    if model_fingerprint(policy)!=policy_before:raise RuntimeError("Frozen policy changed during adapter training")
    atomic_json(output/"summary.json",dict(status="completed" if step==args.steps else "stopped_at_checkpoint",steps=step,planned_steps=args.steps,seconds=time.monotonic()-start,
                policy_unchanged=True,**{k:v for k,v in config.items() if k not in ("steps",)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",default="outputs/real_train/checkpoint");p.add_argument("--data",default="outputs/real_dataset")
    p.add_argument("--styles-data",default="outputs/styles");p.add_argument("--style",type=validate_style_slug,required=True)
    p.add_argument("--output",required=True);p.add_argument("--device",default="auto")
    p.add_argument("--steps",type=int,default=1000);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--grad-accum",type=int,default=2)
    p.add_argument("--rank",type=int,default=16);p.add_argument("--alpha",type=float,default=16.);p.add_argument("--start-block",type=int,default=0)
    p.add_argument("--early-rank",type=int,default=8,help="Rank for blocks 0–3; use --start-block 4 to freeze them")
    p.add_argument("--stop-after",type=int,help="Stop at this optimizer step, retaining the planned total-step schedule")
    p.add_argument("--resume",help="Safe adapter training-state checkpoint directory")
    p.add_argument("--no-unpatch",action="store_true");p.add_argument("--lr",type=float,default=2.5e-4)
    p.add_argument("--warmup",type=int,default=100);p.add_argument("--ema",type=float,default=.995)
    p.add_argument("--eval-every",type=int,default=250);p.add_argument("--eval-items",type=int,default=4);p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.grad_accum,a.rank,a.eval_every,a.eval_items)<1 or not 0<=a.ema<1:p.error("Invalid training bounds")
    train(a)


if __name__=="__main__":main()
