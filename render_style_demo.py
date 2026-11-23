"""Render ONE comparison GIF from held-out real PandaSet data, never from fixtures.

Outputs stay under outputs/ by default. Inspect the trained result before choosing
the one GIF for the README; this command never publishes or edits documentation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from unrealityview.checkpoint import load_checkpoint
from unrealityview.data.cache import atomic_json
from unrealityview.data.dataset import CachedEpisodeChunkDataset,stage_context
from unrealityview.data.splits import assert_unseen_test
from unrealityview.device import select_device,autocast_for
from unrealityview.models.lora import LoRAWeightManager,load_lora,model_fingerprint
from unrealityview.models.pretrained_taesd import PretrainedTAESD
from unrealityview.visuals import to_image


@torch.no_grad()
def render(args):
    device=select_device(args.device)
    model,policy,config=load_checkpoint(args.base,device)
    model.requires_grad_(False);policy.requires_grad_(False)
    before=model_fingerprint(policy);manager=LoRAWeightManager(model)
    data=CachedEpisodeChunkDataset(args.data,"test",jitter=False)
    assert_unseen_test(config.get("data_provenance"),[m["source_id"] for m in data.metadata])
    meta=data.metadata[args.episode]
    if meta.get("source")!="pandaset":raise ValueError("Public demonstration inputs must be real PandaSet")
    adapters={}
    for style in ("comic","cyberpunk"):
        candidates=[Path(args.lora_dir)/f"{style}_lora.safetensors",Path(args.lora_dir)/style/f"{style}_lora.safetensors"]
        path=next((p for p in candidates if p.is_file()),None)
        if path is None:raise ValueError(f"Train and validate the {style} adapter before rendering a style demo")
        adapters[style]=load_lora(path,manager.base_sha256)
    episode=data.episodes[args.episode];decoder=PretrainedTAESD().to(device)
    initial=episode["chase_latents"][:8].permute(1,0,2,3)[None].float().to(device)
    sequences={}
    for style in ("reality","comic","cyberpunk","hybrid"):
        requested=[] if style=="reality" else [(name,adapters[name],.55 if style=="hybrid" else 1.)
                     for name in (("comic","cyberpunk") if style=="hybrid" else (style,))]
        manager.apply_adapters(requested);previous=initial.clone();videos=[]
        generator=torch.Generator(device=device).manual_seed(args.seed)
        for chunk in range(4):
            start=8+chunk*8;ids,ages=stage_context(episode,start,float(meta["fps"]))
            context=episode["uve_tokens"][ids][None].float().to(device)
            with autocast_for(device):
                previous=model.sample_chunk(previous,context,1,ages[None].to(device),
                    noise=torch.randn(previous.shape,device=device,generator=generator))
            videos.append(decoder.decode(previous.float())[0].cpu());previous=previous.float()
        sequences[style]=torch.cat(videos,1)
    manager.restore_base()
    if model_fingerprint(policy)!=before:raise RuntimeError("Planner changed during style rendering")
    cells=[("PandaSet reference",None),("Reality model","reality"),("Comic","comic"),("Cyberpunk","cyberpunk"),("Hybrid","hybrid")]
    frames=[];directory=data.paths[args.episode].parent
    for index in range(32):
        canvas=Image.new("RGB",(640,654),"#111820");draw=ImageDraw.Draw(canvas)
        for cell,(label,style) in enumerate(cells):
            x,y=cell%2*320,cell//2*218;draw.text((x+7,y+7),label,fill="white")
            if style is None:
                with Image.open(directory/"raw/chase"/f"{index+8:05d}.jpg") as source:image=source.convert("RGB")
            else:image=to_image(sequences[style][:,index])
            canvas.paste(image,(x,y+26))
        draw.text((330,450),"UnrealityView / 1 NFE\nReal PandaSet inputs\nIndependent frozen planner\n"+meta.get("target_view","native_front"),fill="white",spacing=10)
        frames.append(canvas)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=1000/float(meta["fps"]),loop=0)
    atomic_json(output.with_suffix(".json"),dict(source_id=meta["source_id"],split="test",frames=32,fps=meta["fps"],
                target_view=meta.get("target_view"),base_step=config["step"],base_sha256=manager.base_sha256,
                styles=list(sequences),policy_unchanged=True,seed=args.seed,
                publication_status="Not published; requires completed quality training and visual inspection"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",default="outputs/real_train/checkpoint");p.add_argument("--data",default="outputs/real_dataset")
    p.add_argument("--lora-dir",default="outputs/adapters");p.add_argument("--output",default="outputs/demo/unrealityview.gif")
    p.add_argument("--episode",type=int,default=0);p.add_argument("--seed",type=int,default=719);p.add_argument("--device",default="auto")
    a=p.parse_args()
    if a.episode<0:p.error("episode index must be nonnegative")
    render(a)


if __name__=="__main__":main()
