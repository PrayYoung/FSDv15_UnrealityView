"""Inventory, freeze scene splits, and cache camera-only Waymo/PandaSet sequences."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from unrealityview.data.cache import episode_grid,atomic_json,save_episode,resolve_safe_subpath,load_episode
from unrealityview.data.real_world import CAMERA_SLOTS,PandaSetSequence,WaymoSequence
from unrealityview.data.splits import stable_split,validate_split_manifest
from unrealityview.device import select_device,autocast_for
from unrealityview.models.mini_uve import MiniUVE
from unrealityview.models.pretrained_taesd import PretrainedTAESD


def plan(pandaset_root=None,waymo_root=None,seed=42):
    sequences=[]
    if pandaset_root:
        root=Path(pandaset_root).resolve()
        for sequence in sorted(root.iterdir()):
            if (sequence/"camera/front_camera/timestamps.json").is_file():
                sequences.append(dict(dataset="pandaset",sequence=sequence.name,root=str(root),
                                      split=stable_split("pandaset",sequence.name,seed)))
    if waymo_root:
        for partition in ("training","validation"):
            root=Path(waymo_root).resolve()/partition
            for path in sorted((root/"camera_image").glob("*.parquet")):
                split="test" if partition=="validation" else stable_split("waymo",path.stem,seed)
                # Reserve training-context test hash buckets as validation;
                # final Waymo testing is always the official validation partition.
                if partition=="training" and split=="test":split="val"
                sequences.append(dict(dataset="waymo",sequence=path.stem,root=str(root),
                                      split=split,official_partition=partition))
    if not sequences:raise ValueError("No native camera sequences found under the supplied roots")
    return validate_split_manifest(dict(version=1,seed=seed,source="Waymo + PandaSet camera-only",sequences=sequences))


def target_frame(targets,entry,index,timestamp):
    """External elevated-view teacher contract; never relabel front images."""
    directory=resolve_safe_subpath(targets,f"{entry['dataset']}/{entry['sequence']}")
    metadata=json.loads((directory/"manifest.json").read_text())
    if metadata.get("target_view")!="elevated_chase":raise ValueError("Teacher must explicitly declare elevated_chase")
    if abs(float(metadata["timestamps"][index])-timestamp)>.001:raise ValueError("Teacher/image timestamp mismatch")
    expected=dict(camera_back=6.8,camera_height=3.8,pitch_deg=19.,fov=75.)
    if any(abs(float(metadata["camera"][k])-v)>1e-5 for k,v in expected.items()):
        raise ValueError("Teacher must match the cockpit chase calibration")
    if metadata.get("source_id")!=f"{entry['dataset']}/{entry['sequence']}":raise ValueError("Teacher source identity mismatch")
    path=resolve_safe_subpath(directory,metadata["frames"][index])
    with Image.open(path) as image:return image.convert("RGB").resize((320,192)),metadata.get("teacher","external teacher")


@torch.no_grad()
def cache_manifest(manifest,output,device="auto",max_frames=None,target="front",targets=None,encoder=None,decoder=None,reuse_existing=False,uve_grid=(4,8)):
    manifest=validate_split_manifest(manifest)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/"split_manifest.json").exists():
        prior=json.loads((output/"split_manifest.json").read_text())
        if prior!=manifest:raise ValueError("Output already has a different frozen split manifest; use a new output")
    atomic_json(output/"split_manifest.json",manifest)
    device=select_device(device)
    encoder=MiniUVE(token_grid=uve_grid).to(device).eval() if encoder is None else encoder
    decoder=PretrainedTAESD().to(device).eval() if decoder is None else decoder
    provenance=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    names={"pandaset":"PandaSet","waymo":"Waymo Perception v2"}
    source=" + ".join(names[k] for k in sorted({e["dataset"] for e in manifest["sequences"]}))+" camera-only"
    desired_grid=tuple(getattr(encoder,"token_grid",(4,8)))
    if (output/'manifest.json').exists():
        prior=json.loads((output/'manifest.json').read_text())
        if tuple(prior.get('uve_grid',(4,8)))!=desired_grid:raise ValueError('Existing manifest UVE grid mismatch; use a new output')
    atomic_json(output/"manifest.json",dict(source=source,target_view="native_front" if target=="front" else "elevated_chase",
                 uve_grid=list(getattr(encoder,"token_grid",(4,8))),split_manifest_sha256=provenance,temporal_policy="native timestamps; no invented 36Hz frames"))
    for entry in manifest["sequences"]:
        source_id=f"{entry['dataset']}/{entry['sequence']}"
        name=entry['dataset']+'_'+hashlib.sha256(source_id.encode()).hexdigest()[:16]
        directory=output/entry['split']/name
        if reuse_existing and (directory/'episode.safetensors').is_file():
            meta=json.loads((directory/'metadata.json').read_text())
            expected_frames=manifest.get('download',{}).get('frames')
            if (meta.get('source_id')!=source_id or meta.get('split')!=entry['split']
                or meta.get('target_view')!=('native_front' if target=='front' else 'elevated_chase')
                or meta.get('split_manifest_sha256')!=provenance or meta['frames']!=expected_frames
                or max_frames is not None and meta['frames']>max_frames):raise ValueError('Existing cache provenance/configuration mismatch')
            existing_tensors=load_episode(directory/'episode.safetensors')
            if episode_grid(existing_tensors,meta)!=tuple(getattr(encoder,'token_grid',(4,8))):raise ValueError('Existing cache UVE grid mismatch')
            print(f'REUSED_CACHE {source_id}',flush=True)
            continue
        sequence=(PandaSetSequence if entry["dataset"]=="pandaset" else WaymoSequence)(entry["root"],entry["sequence"])
        count=len(sequence) if max_frames is None else min(len(sequence),max_frames)
        if count<48:raise ValueError(f"Need >=48 frames for chunks, waypoints and four-chunk rollout: {source_id}")
        name=entry["dataset"]+"_"+hashlib.sha256(source_id.encode()).hexdigest()[:16]
        directory=output/entry["split"]/name
        if (directory/"episode.safetensors").exists():
            existing=json.loads((directory/"metadata.json").read_text())
            if episode_grid(load_episode(directory/"episode.safetensors"),existing)!=tuple(getattr(encoder,"token_grid",(4,8))):raise ValueError("Existing cache UVE grid mismatch")
            if existing.get("target_view")!=("native_front" if target=="front" else "elevated_chase") or existing["frames"]!=count:
                raise ValueError("Existing cache has incompatible target or frame count")
            continue
        tokens,latents,poses,times,masks,ks,extrinsics,cam_times=[],[],[],[],[],[],[],[]
        storage=("front_left","front","front_right","rear_left","rear","rear_right")
        for camera in (*storage,"chase"):(directory/"raw"/camera).mkdir(parents=True,exist_ok=True)
        for index in range(count):
            frame=sequence.frame(index)
            if target=="front":
                target_image=frame.images[1].resize((320,192),Image.Resampling.BILINEAR)
                teacher="observed native front camera; camera-conditioned reconstruction baseline"
            else:
                if targets is None:raise ValueError("--target teacher requires --targets with calibrated novel-view exports")
                target_image,teacher=target_frame(targets,entry,index,frame.timestamp)
            for name,image in zip(storage,frame.images):image.save(directory/"raw"/name/f"{index:05d}.jpg",quality=90)
            target_image.save(directory/"raw/chase"/f"{index:05d}.jpg",quality=90)
            if index==0:
                grid=Image.new("RGB",(960,660),"#101820");draw=ImageDraw.Draw(grid)
                for j,image in enumerate([*frame.images,target_image]):
                    x,y=j%3*320,j//3*220
                    label=CAMERA_SLOTS[j] if j<6 else ("NATIVE FRONT TARGET" if target=="front" else "TEACHER CHASE TARGET")
                    if j<6 and not frame.camera_mask[j]:label+=" / UNAVAILABLE"
                    grid.paste(image.resize((320,192)),(x,y+24));draw.text((x+7,y+7),label,fill="white")
                grid.save(directory/"raw_preview.png")
            cameras=torch.from_numpy(np.stack([np.array(x) for x in frame.images])).permute(0,3,1,2)[None].float().to(device)/255
            video=torch.from_numpy(np.array(target_image)).permute(2,0,1)[None,:,None].float().to(device)/127.5-1
            k=torch.from_numpy(frame.intrinsics).float()[None].to(device)
            ext=torch.from_numpy(frame.camera_to_ego).float()[None].to(device)
            mask=torch.from_numpy(frame.camera_mask)[None].to(device)
            with autocast_for(device):
                tokens.append(encoder(cameras,k,ext,camera_mask=mask)[0].half().cpu())
            latents.append(decoder.encode(video)[0,:,0].half().cpu())
            poses.append(frame.ego_pose);times.append(frame.timestamp);masks.append(frame.camera_mask)
            ks.append(frame.intrinsics);extrinsics.append(frame.camera_to_ego);cam_times.append(frame.camera_timestamps)
        fps=float(1/np.median(np.diff(times)))
        save_episode(directory,dict(uve_tokens=torch.stack(tokens),chase_latents=torch.stack(latents),
                     ego_states=torch.tensor(np.stack(poses),dtype=torch.float32),timestamps=torch.tensor(times,dtype=torch.float64),
                     camera_timestamps=torch.tensor(np.stack(cam_times),dtype=torch.float64),camera_mask=torch.tensor(np.stack(masks)),
                     intrinsics=torch.tensor(np.stack(ks),dtype=torch.float32),camera_to_ego=torch.tensor(np.stack(extrinsics),dtype=torch.float32)),
                     dict(source=entry["dataset"],source_id=source_id,sequence=entry["sequence"],split=entry["split"],frames=count,
                          uve_grid=list(getattr(encoder,"token_grid",(4,8))),fps=fps,target_view="native_front" if target=="front" else "elevated_chase",target_reference=teacher,
                          camera_names=CAMERA_SLOTS,ego_reference=sequence.reference_kind,split_manifest_sha256=provenance,
                          timing="native; no temporal upsampling",pixel_size=[320,192],input_size=[224,140]))
        print(f"CACHED {source_id} split={entry['split']} frames={count} fps={fps:.3f} target={target}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    inventory=sub.add_parser("plan");inventory.add_argument("--pandaset-root");inventory.add_argument("--waymo-root")
    inventory.add_argument("--seed",type=int,default=42);inventory.add_argument("--output",default="outputs/real_splits.json")
    cache=sub.add_parser("cache");cache.add_argument("--manifest",required=True);cache.add_argument("--output",default="outputs/real_dataset")
    cache.add_argument("--device",default="auto");cache.add_argument("--max-frames",type=int)
    cache.add_argument("--reuse-existing",action="store_true")
    cache.add_argument("--uve-grid",type=int,nargs=2,default=(4,8))
    cache.add_argument("--target",choices=("front","teacher"),default="front");cache.add_argument("--targets")
    a=p.parse_args()
    if a.command=="plan":atomic_json(a.output,plan(a.pandaset_root,a.waymo_root,a.seed))
    else:cache_manifest(json.loads(Path(a.manifest).read_text()),a.output,a.device,a.max_frames,a.target,a.targets,reuse_existing=a.reuse_existing,uve_grid=tuple(a.uve_grid))


if __name__=="__main__":main()
