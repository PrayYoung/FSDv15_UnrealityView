"""Held-out metrics, identical-noise NFE comparison, and autoregressive rollout."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from fsd_reality.checkpoint import load_checkpoint
from fsd_reality.data.dataset import CachedEpisodeChunkDataset
from fsd_reality.data.cache import atomic_json
from fsd_reality.device import select_device,autocast_for
from fsd_reality.models.pretrained_taesd import PretrainedTAESD
from fsd_reality.visuals import contact_sheet,save_rollout


def pixel_metrics(prediction, target):
    prediction,target = prediction.float().clamp(-1,1),target.float().clamp(-1,1)
    mse = (prediction-target).square().mean().item()
    return {"psnr_db":float(10*np.log10(4/max(mse,1e-12))),
            "pixel_l1":(prediction-target).abs().mean().item()}


@torch.no_grad()
def evaluate_models(model, policy, decoder, dataset, device, output, max_items=8, seed=917):
    model.eval();policy.eval()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results={str(n):[] for n in (1,2,4)}
    for i in range(min(max_items,len(dataset))):
        item=dataset[i]
        previous=item["z_prev"][None].to(device)
        target=item["z0"][None].to(device)
        tokens=item["uve_tokens"][None].to(device)
        ages=item["stage_ages"][None].to(device)
        # Raw chase JPEGs are the pixel target. TAESD reconstruction metrics are
        # also reported separately, so lossy cache compression cannot inflate PSNR.
        from PIL import Image
        raw=dataset.paths[item["episode"]].parent/"raw"/"chase"
        arrays=[np.array(Image.open(raw/f"{item['start']+j:05d}.jpg").convert("RGB")) for j in range(8)]
        truth=torch.from_numpy(np.stack(arrays)).permute(3,0,1,2)[None].to(device).float()/127.5-1
        reconstruction=decoder.decode(target)
        gen=torch.Generator(device=device).manual_seed(seed+i)
        noise=torch.randn(previous.shape,device=device,generator=gen)
        predictions={}
        for n in (1,2,4):
            with autocast_for(device):
                latent=model.sample_chunk(previous,tokens,n,ages,noise=noise)
                waypoints=policy(tokens[:,0])
            pixels=decoder.decode(latent.float())
            predictions[n]=pixels
            results[str(n)].append({**pixel_metrics(pixels,truth),
                   "latent_l1":(latent-target).abs().mean().item(),
                   "waypoint_l1_m":(waypoints[0,:,:2]-item["waypoints"][:,:2].to(device)).abs().mean().item(),
                   "episode":item["episode"],"start":item["start"]})
        if i==0:
            contact_sheet([("Previous chunk",decoder.decode(previous)[0]),("Ground truth",truth[0]),
                           ("1-NFE",predictions[1][0]),("2-NFE",predictions[2][0])],output/"contact_sheet.png")
            contact_sheet([("Ground truth",truth[0]),("TAESD reconstruction",reconstruction[0]),
                           ("4-NFE",predictions[4][0])],output/"reconstruction.png")
    means={n:{key:float(np.mean([r[key] for r in rows])) for key in
               ("psnr_db","pixel_l1","latent_l1","waypoint_l1_m")} for n,rows in results.items()}
    manifest=dataset.root/"manifest.json"
    source=json.loads(manifest.read_text()).get("source","unknown") if manifest.is_file() else "unknown"
    report={"split":"val","dataset_source":source,"samples":min(max_items,len(dataset)),"seed":seed,"nfe":means,"per_chunk":results,
            "pixel_target":"raw chase JPEG; pixels normalized [-1,1]",
            "note":"Offline video rollout; not a vehicle-control or safety benchmark"}
    atomic_json(output/"metrics.json",report)
    # Four consecutive 8-frame chunks, never reseeding previous with GT.
    ep=dataset.episodes[0]
    previous=ep["chase_latents"][:8].permute(1,0,2,3)[None].float().to(device)
    videos,drift=[],[]
    for chunk in range(4):
        start=8+chunk*8
        ids=(start+torch.tensor([-1,1,3,5])).clamp(max=len(ep["uve_tokens"])-1)
        context=ep["uve_tokens"][ids][None].float().to(device)
        with autocast_for(device):
            previous=model.sample_chunk(previous,context,1,noise=torch.randn(previous.shape,device=device,generator=gen))
        pixels=decoder.decode(previous.float())
        videos.append(pixels[0])
        gt=ep["chase_latents"][start:start+8].permute(1,0,2,3)[None].float().to(device)
        drift.append({"chunk":chunk,"latent_l1":(previous-gt).abs().mean().item()})
        previous=previous.float()
    rollout=torch.cat(videos,dim=1)
    save_rollout(rollout,output/"closed_loop_32_frames.gif")
    contact_sheet([("Closed-loop 1-NFE",rollout)],output/"closed_loop_contact_sheet.png",(0,7,15,23,31))
    atomic_json(output/"rollout_metrics.json",{"chunks":drift,"zero_drift_verified":False,
                                              "note":"Drift is reported; no zero-drift assertion"})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",default="outputs/train/checkpoint")
    p.add_argument("--data",default="outputs/dataset")
    p.add_argument("--output",default="outputs/evaluation")
    p.add_argument("--device",default="auto")
    p.add_argument("--max-items",type=int,default=8)
    a=p.parse_args();device=select_device(a.device)
    model,policy,_=load_checkpoint(a.checkpoint,device)
    decoder=PretrainedTAESD().to(device)
    dataset=CachedEpisodeChunkDataset(a.data,"val",jitter=False)
    print(json.dumps(evaluate_models(model,policy,decoder,dataset,device,a.output,a.max_items),indent=2))


if __name__=="__main__":
    main()
