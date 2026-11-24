"""Re-encode camera features from cached JPEGs; never modify source tensors/RGB."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch
import numpy as np
from PIL import Image
from unrealityview.models.mini_uve import MiniUVE
from unrealityview.data.cache import load_episode,save_episode,atomic_json,episode_grid
from unrealityview.device import select_device,autocast_for

SLOTS=('front_left','front','front_right','rear_left','rear','rear_right')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


@torch.no_grad()
def encode_episode(path,encoder,device,batch_size=8):
    path=Path(path);old=load_episode(path);outputs=[]
    for start in range(0,len(old['uve_tokens']),batch_size):
        stop=min(start+batch_size,len(old['uve_tokens']))
        arrays=[]
        for i in range(start,stop):
            cameras=[]
            for name in SLOTS:
                with Image.open(path.parent/'raw'/name/f'{i:05d}.jpg') as im:
                    if im.size!=(224,140):raise ValueError('Cached camera size/order contract requires 224x140 JPEGs')
                    cameras.append(np.array(im.convert('RGB')))
            arrays.append(cameras)
        cameras=torch.from_numpy(np.array(arrays)).permute(0,1,4,2,3).float().to(device)/255
        args=[old[k][start:stop].to(device) for k in ('intrinsics','camera_to_ego','camera_mask')]
        with autocast_for(device):tokens=encoder(cameras,args[0],args[1],camera_mask=args[2])
        outputs.append(tokens.half().cpu())
    return {**old,'uve_tokens':torch.cat(outputs)}


def parity(original,reencoded):
    a=original['uve_tokens'].float();b=reencoded['uve_tokens'].float()
    if a.shape!=b.shape:raise ValueError('Parity requires identical 4x8 grids')
    slots=[]
    for i in range(6):
        mask=original['camera_mask'][:,i].bool();x=a.reshape(len(a),6,-1,384)[mask,i].flatten();y=b.reshape(len(b),6,-1,384)[mask,i].flatten()
        if not mask.any():continue
        relative=((x-y).square().mean().sqrt()/x.square().mean().sqrt().clamp_min(1e-8)).item()
        cosine=torch.nn.functional.cosine_similarity(x[None],y[None]).item()
        slots.append(dict(slot=SLOTS[i],relative_rms=relative,cosine=cosine))
    return {'passed':bool(slots) and all(x['relative_rms']<=.20 and x['cosine']>=.98 for x in slots),
        'thresholds':{'relative_rms_max':.20,'cosine_min':.98},'per_camera':slots,
        'note':'Nonzero tolerance accommodates re-encoding quality90 JPEGs instead of original decoded pixels; not exact-token equivalence.'}


def link_or_copy(source,target):
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)


def reencode(source,output,grid,device,report,encoder=None,parity_encoder=None):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output or source in output.parents or output.exists():raise ValueError('Use a new output outside source cache')
    paths=sorted((source/'train').glob('*/episode.safetensors'))
    if not paths:raise ValueError('Training cache missing')
    for p in source.glob('*/*/episode.safetensors'):
        meta=json.loads(p.with_name('metadata.json').read_text());old=load_episode(p)
        if episode_grid(old,meta)!=(4,8):raise ValueError('Ablation source must be legacy4x8')
        if meta.get('cache_tensor_sha256') and digest(p)!=meta['cache_tensor_sha256']:raise ValueError('Source tensor fingerprint mismatch')
    check=parity_encoder if parity_encoder is not None else MiniUVE(token_grid=(4,8)).to(device).eval()
    check_result=parity(load_episode(paths[0]),encode_episode(paths[0],check,device))
    atomic_json(report,check_result)
    if not check_result['passed']:raise ValueError('4x8 JPEG parity failed; no new grid cache or training permitted')
    del check
    encoder=encoder if encoder is not None else MiniUVE(token_grid=grid).to(device).eval()
    output.mkdir(parents=True)
    for name in ('split_manifest.json','manifest.json'):
        m=json.loads((source/name).read_text())
        # Frozen split manifest is copied exactly; grid provenance lives elsewhere.
        if name=='manifest.json':m.update(uve_grid=list(grid),reencoded_from_cached_raw_jpeg=True)
        atomic_json(output/name,m)
    for path in sorted(source.glob('*/*/episode.safetensors')):
        target=output/path.relative_to(source);target.parent.mkdir(parents=True)
        tensors=encode_episode(path,encoder,device)
        shutil.copytree(path.parent/'raw',target.parent/'raw',copy_function=link_or_copy)
        meta=json.loads(path.with_name('metadata.json').read_text())
        meta.update(uve_grid=list(grid),reencoded_from_cached_raw_jpeg=True,source_episode_sha256=digest(path))
        meta.pop('cache_tensor_sha256',None)
        save_episode(target.parent,tensors,meta)
        meta['cache_tensor_sha256']=digest(target);atomic_json(target.with_name('metadata.json'),meta)
        print('REENCODED',meta['source_id'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--grid',type=int,nargs=2,default=(10,16));p.add_argument('--device',default='cuda');p.add_argument('--parity-report',required=True);a=p.parse_args()
    reencode(a.source,a.output,tuple(a.grid),select_device(a.device),a.parity_report)


if __name__=='__main__':main()
