"""Make a labelled PROCEDURAL cache to exercise the pipeline without simulator EGL.

This is an integration fixture, not MetaDrive data or evidence of driving quality.
The raw images are saved alongside real pretrained DINOv2 tokens and TAESD latents.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from app.demo import demo_frame
from collect_metadrive import CAMERAS
from fsd_reality.data.cache import save_episode,atomic_json
from fsd_reality.device import select_device,autocast_for
from fsd_reality.models.mini_uve import MiniUVE
from fsd_reality.models.pretrained_taesd import PretrainedTAESD


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="outputs/procedural_smoke/dataset")
    p.add_argument("--frames",type=int,default=80)
    p.add_argument("--device",default="auto")
    a=p.parse_args()
    if a.frames<64:p.error("frames must be >=64")
    root=Path(a.output);device=select_device(a.device)
    uve,vae=MiniUVE().to(device).eval(),PretrainedTAESD().to(device).eval()
    meta={"source":"procedural integration fixture","fps":36,"frames":a.frames,
          "purpose":"Integration verification only; not a driving-quality dataset"}
    atomic_json(root/"manifest.json",meta)
    for ep,split in enumerate(("train","train","val")):
        directory=root/split/f"episode_{ep:04d}"
        if (directory/"episode.safetensors").is_file():continue
        for name in (*CAMERAS,"chase"):(directory/"raw"/name).mkdir(parents=True,exist_ok=True)
        tokens,latents=[],[]
        for frame in range(a.frames):
            images=[demo_frame(frame+ep*83,c).resize((224,140)) for c in range(6)]
            images.append(demo_frame(frame+ep*83))
            for name,image in zip((*CAMERAS,"chase"),images):image.save(directory/"raw"/name/f"{frame:05d}.jpg",quality=90)
            if frame==0:
                from PIL import Image,ImageDraw
                grid=Image.new("RGB",(960,660),"#111820");draw=ImageDraw.Draw(grid)
                for i,(name,image) in enumerate(zip((*CAMERAS,"chase"),images)):
                    x,y=i%3*320,i//3*220;draw.text((x+6,y+5),name+" / PROCEDURAL",fill="white");grid.paste(image.resize((320,192)),(x,y+24))
                grid.save(directory/"raw_preview.png")
            camera=torch.from_numpy(np.stack([np.array(i) for i in images[:6]])).permute(0,3,1,2)[None].float().to(device)/255
            chase=torch.from_numpy(np.array(images[6])).permute(2,0,1)[None,:,None].float().to(device)/127.5-1
            with autocast_for(device):
                tokens.append(uve(camera)[0].half().cpu())
                latents.append(vae.encode(chase)[0,:,0].half().cpu())
        poses=torch.zeros(a.frames,3);poses[:,0]=torch.arange(a.frames)/36*10
        save_episode(directory,{"uve_tokens":torch.stack(tokens),"chase_latents":torch.stack(latents),"ego_states":poses},
                     {**meta,"split":split,"episode":ep,"ego_pose_kind":"analytical fixture; not simulated motion"})
        print(f"PROCEDURAL_CACHE {directory}",flush=True)


if __name__=="__main__":main()
