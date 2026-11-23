"""Deterministic color/ink teachers with optical-flow-based temporal stabilization.

This is a 2D real-video baseline, not a 3D-consistent stylized Gaussian teacher.
Flow consistency rejects disocclusions; it cannot guarantee zero flicker.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.utils.data import Dataset
from safetensors.torch import load_file,save_file
from unrealityview.data.cache import atomic_json
from unrealityview.data.dataset import CachedEpisodeChunkDataset
from unrealityview.device import select_device,autocast_for
from unrealityview.models.pretrained_taesd import PretrainedTAESD

STYLES=("comic","cyberpunk")


def spatial_shader(image,style,coordinates=None):
    import cv2
    if style not in STYLES:raise ValueError("Unknown shader style")
    rgb=np.asarray(image,dtype=np.uint8)
    x=rgb.astype(np.float32)/255
    h,w=x.shape[:2];y=np.linspace(0,1,h,dtype=np.float32)[:,None,None]
    edge=np.sqrt(cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3)**2+cv2.Sobel(x,cv2.CV_32F,0,1,ksize=3)**2).max(-1)
    if style=="comic":
        smooth=rgb
        for _ in range(3):smooth=cv2.bilateralFilter(smooth,5,35,5)
        x=smooth.astype(np.float32)/255
        luminance=x.mean(-1,keepdims=True);x=np.clip(luminance+(x-luminance)*1.65,0,1)
        # Smooth five-level posterization, not randomized framewise diffusion.
        levels=np.linspace(.125,.875,4,dtype=np.float32)
        x=(1/(1+np.exp(np.clip(-(x[...,None]-levels)*45,-80,80)))).sum(-1)/4
        if coordinates is None:
            gy,gx=np.mgrid[:h,:w].astype(np.float32);coordinates=np.stack((gx,gy),-1)
        angle=(coordinates[...,0]+coordinates[...,1])/np.sqrt(2)
        perpendicular=(coordinates[...,0]-coordinates[...,1])/np.sqrt(2)
        dots=((angle%7-3.5)**2+(perpendicular%7-3.5)**2)<1.8
        mid=np.clip(1-abs(luminance[...,0]-.5)*2,0,1)
        x*=1-(dots*mid*.22)[...,None]
        ink=np.clip((edge-.12)*1.8,0,.8)[...,None]
        return np.clip(x*(1-ink),0,1)
    # Color and vertical heuristics, explicitly not semantic segmentation.
    blue=np.clip((x[...,2:3]-x[...,:1]+.12)*3,0,1)
    sky=blue*np.clip((.64-y)*5,0,1)
    road=np.clip((y-.42)*3,0,1)*(1-np.clip(x.max(-1,keepdims=True)-x.min(-1,keepdims=True),0,1))
    midnight=np.array([.055,.035,.16],np.float32)
    horizon=np.exp(-((y-.48)/.15)**2)*np.array([.28,.015,.24],np.float32)
    color=x*.34+np.array([.025,.045,.08],np.float32)
    color=color*(1-sky)+(midnight+horizon)*sky
    color=color*(1-road*.6)+np.array([.015,.035,.06],np.float32)*road*.6
    neon=np.clip((edge-.09)*1.7,0,1)[...,None]
    cyan=np.array([.03,.85,1.],np.float32);magenta=np.array([1.,.025,.6],np.float32)
    palette=cyan*(1-y)+magenta*y
    glow=cv2.GaussianBlur(neon,(0,0),1.8)[...,None] if neon.ndim==2 else cv2.GaussianBlur(neon,(0,0),1.8)
    if glow.ndim==2:glow=glow[...,None]
    color+=palette*(neon*.65+glow*.8)
    return np.clip(color,0,1)


class TemporalStylizer:
    def __init__(self,style,temporal_strength=.65):
        if style not in STYLES or not 0<=temporal_strength<=1:raise ValueError("Invalid temporal shader configuration")
        self.style=style;self.strength=temporal_strength
        self.previous=None;self.previous_style=None;self.coordinates=None
        self.flow_errors=[]

    def reset(self):
        self.previous=None;self.previous_style=None;self.coordinates=None;self.flow_errors=[]

    def __call__(self,image):
        import cv2
        rgb=np.asarray(image,dtype=np.uint8);h,w=rgb.shape[:2]
        gy,gx=np.mgrid[:h,:w].astype(np.float32);grid=np.stack((gx,gy),-1)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        confidence=None;warped=None;coordinates=grid
        if self.previous is not None:
            if self.previous.shape!=rgb.shape:raise ValueError("Frames in a sequence must have fixed dimensions")
            prior=cv2.cvtColor(self.previous,cv2.COLOR_RGB2GRAY)
            backward=cv2.calcOpticalFlowFarneback(gray,prior,None,.5,3,15,3,5,1.2,0)
            forward=cv2.calcOpticalFlowFarneback(prior,gray,None,.5,3,15,3,5,1.2,0)
            mapping=grid+backward
            warp=lambda array:cv2.remap(array,mapping,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            consistency=np.linalg.norm(backward+warp(forward),axis=-1)
            photometric=np.abs(rgb.astype(np.float32)/255-warp(self.previous.astype(np.float32)/255)).mean(-1)
            inside=(mapping[...,0]>=0)&(mapping[...,0]<w-1)&(mapping[...,1]>=0)&(mapping[...,1]<h-1)
            confidence=np.exp(-(consistency/1.5)**2)*(photometric<.12)*inside
            coordinates=warp(self.coordinates)
            coordinates=np.where((confidence>.5)[...,None],coordinates,grid)
            warped=warp(self.previous_style)
            self.flow_errors.append(float(np.mean(consistency)))
        styled=spatial_shader(rgb,self.style,coordinates)
        if confidence is not None:
            amount=(confidence*self.strength)[...,None]
            styled=styled*(1-amount)+warped*amount
        self.previous=rgb.copy();self.previous_style=styled;self.coordinates=coordinates
        return np.rint(styled*255).clip(0,255).astype(np.uint8)


@torch.no_grad()
def build_style_cache(base_root,output,styles=STYLES,device="auto",decoder=None):
    base_root=Path(base_root).resolve();output=Path(output).resolve()
    if output==base_root or base_root in output.parents:raise ValueError("Keep style outputs separate from the immutable base cache")
    device=select_device(device);decoder=PretrainedTAESD().to(device) if decoder is None else decoder
    # Instantiating all splits also audits source leakage; do not style across sequence boundaries.
    for split in ("train","val","test"):
        if not any((base_root/split).glob("*/episode.safetensors")):continue
        data=CachedEpisodeChunkDataset(base_root,split,jitter=False)
        for path,episode,meta in zip(data.paths,data.episodes,data.metadata):
            if meta.get("source")!="pandaset":raise ValueError("Style cache requires real PandaSet camera data")
            base_sha=hashlib.sha256(path.read_bytes()).hexdigest()
            for style in styles:
                stylizer=TemporalStylizer(style);latents=[];samples=[]
                directory=output/style/split/path.parent.name;directory.mkdir(parents=True,exist_ok=True)
                (directory/"frames").mkdir(exist_ok=True)
                for index in range(len(episode["chase_latents"])):
                    with Image.open(path.parent/"raw/chase"/f"{index:05d}.jpg") as image:base=image.convert("RGB")
                    styled=Image.fromarray(stylizer(base))
                    styled.save(directory/"frames"/f"{index:05d}.png")
                    tensor=torch.from_numpy(np.array(styled)).permute(2,0,1)[None,:,None].float().to(device)/127.5-1
                    with autocast_for(device):z=decoder.encode(tensor)
                    latents.append(z[0,:,0].half().cpu())
                    if index in (0,len(episode["chase_latents"] )//2,len(episode["chase_latents"])-1):
                        reconstruction=decoder.decode(z.float())[0,:,0].float().cpu().clamp(-1,1)
                        reconstruction=Image.fromarray(((reconstruction.permute(1,2,0).numpy()+1)*127.5).astype(np.uint8))
                        samples.append((base,styled,reconstruction))
                temporary=directory/"style.safetensors.tmp"
                save_file({"style_latents":torch.stack(latents).contiguous()},str(temporary),metadata={"format":"unrealityview-style-cache-v1"})
                temporary.replace(directory/"style.safetensors")
                atomic_json(directory/"metadata.json",dict(style=style,source_id=meta.get("source_id"),split=split,
                       base_sha256=base_sha,frames=len(latents),teacher="deterministic shader + bidirectional optical-flow confidence",
                       reference="raw PandaSet target JPEG",target_view=meta.get("target_view"),fps=meta.get("fps"),
                       flow_consistency_px_mean=float(np.mean(stylizer.flow_errors)) if stylizer.flow_errors else 0.,
                       limitation="2D temporal baseline; no 3D-consistency or zero-flicker guarantee"))
                w,h=samples[0][0].size;grid=Image.new("RGB",(len(samples)*w,3*(h+24)),"#111820");draw=ImageDraw.Draw(grid)
                for col,sample in enumerate(samples):
                    for row,(label,image) in enumerate(zip(("Real reference","Style teacher","TAESD round trip"),sample)):
                        y=row*(h+24);draw.text((col*w+7,y+6),label,fill="white");grid.paste(image,(col*w,y+24))
                grid.save(directory/"sanity.png")
                print(f"STYLE_CACHE {style} {split} {meta.get('source_id')}",flush=True)


class StylizedChunkDataset(Dataset):
    def __init__(self,base_root,style_root,style,split="train",real_previous_ratio=.2):
        if style not in STYLES or not 0<=real_previous_ratio<=1:raise ValueError("Invalid style dataset configuration")
        self.base=CachedEpisodeChunkDataset(base_root,split,jitter=split=="train")
        self.real_previous_ratio=real_previous_ratio;self.styles=[];self.directories=[]
        for path,episode in zip(self.base.paths,self.base.episodes):
            directory=Path(style_root)/style/split/path.parent.name
            metadata=json.loads((directory/"metadata.json").read_text())
            if metadata["base_sha256"]!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError("Style cache base fingerprint mismatch")
            if metadata["style"]!=style or metadata["split"]!=split:raise ValueError("Style cache identity mismatch")
            z=load_file(str(directory/"style.safetensors"))["style_latents"]
            if z.shape!=episode["chase_latents"].shape or not torch.isfinite(z).all():raise ValueError("Invalid styled latents")
            self.styles.append(z)
            self.directories.append(directory)

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        item=self.base[index];episode,start=self.base.index[index];z=self.styles[episode]
        item["z0"]=z[start:start+8].permute(1,0,2,3).float()
        use_real=torch.rand(()).item()<self.real_previous_ratio
        if not use_real:item["z_prev"]=z[start-8:start].permute(1,0,2,3).float()
        item["previous_is_real"]=use_real
        return item
