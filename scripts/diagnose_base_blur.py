"""Inference-only matched blur diagnosis; frozen evaluation schemas stay unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from evaluate import pixel_metrics
from unrealityview.checkpoint import load_checkpoint
from unrealityview.data.cache import atomic_json,require_uve_grid
from unrealityview.data.dataset import CachedEpisodeChunkDataset
from unrealityview.device import autocast_for,select_device
from unrealityview.models.pretrained_taesd import PretrainedTAESD
from unrealityview.visuals import contact_sheet


def select_indices(dataset,count=40):
    if dataset.split=='val':
        # Matches evaluate_models for the single-source PandaSet benchmark.
        indices=list(range(min(count,len(dataset))))
    elif dataset.split=='train':
        indices=[];seen=set()
        for i,(ep,_) in enumerate(dataset.index):
            if ep not in seen:indices.append(i);seen.add(ep)
            if len(indices)==count:break
    else:raise ValueError('Reserved test scenes are excluded')
    if len(indices)!=count:raise ValueError('Insufficient deterministic windows/scenes')
    return indices


def donor_indices(dataset,indices):
    donors=[]
    for i,index in enumerate(indices):
        identity=dataset.metadata[dataset.index[index][0]]['source_id']
        for offset in range(1,len(indices)):
            other=indices[(i+offset)%len(indices)]
            if dataset.metadata[dataset.index[other][0]]['source_id']!=identity:
                donors.append(other);break
        else:raise ValueError('A different-scene conditioning donor is required')
    return donors


def measurements(latent,target,pixels,truth):
    if not all(torch.isfinite(x).all() for x in (latent,target,pixels,truth)):
        raise ValueError('Nonfinite diagnostic prediction')
    def one(z,t,p,y):return {**pixel_metrics(p,y),'latent_l1':(z.float()-t.float()).abs().mean().item()}
    result=one(latent,target,pixels,truth)
    result['per_frame']=[one(latent[:,:,j:j+1],target[:,:,j:j+1],pixels[:,:,j:j+1],truth[:,:,j:j+1]) for j in range(target.shape[2])]
    return result


def aggregate(rows):
    keys=('psnr_db','pixel_l1','latent_l1')
    return {**{k:float(np.mean([row[k] for row in rows])) for k in keys},
        'per_frame':[{k:float(np.mean([row['per_frame'][j][k] for row in rows])) for k in keys} for j in range(len(rows[0]['per_frame']))]}


@torch.no_grad()
def diagnose_split(model,decoder,dataset,device,output,count=40,seed=917):
    model.eval();indices=select_indices(dataset,count);donors=donor_indices(dataset,indices)
    results={str(n):[] for n in (1,2,4)}
    controls={name:[] for name in ('swapped_uve','frozen_stage_uve','zero_z_prev','swapped_z_prev')} if dataset.split=='val' else {}
    baselines={'taesd_roundtrip':[],'prev_chunk_copy':[]};windows=[];visual=[];shown=set()
    for i,(index,donor) in enumerate(zip(indices,donors)):
        item=dataset[index];other=dataset[donor]
        previous=item['z_prev'][None].to(device);target=item['z0'][None].to(device)
        tokens=item['uve_tokens'][None].to(device);ages=item['stage_ages'][None].to(device)
        if not tokens.ne(0).any(-1).any(-1).all():raise ValueError('Every stage must have valid camera tokens')
        raw=dataset.paths[item['episode']].parent/'raw/chase'
        frames=[np.array(Image.open(raw/f"{item['start']+j:05d}.jpg").convert('RGB')) for j in range(8)]
        truth=torch.from_numpy(np.stack(frames)).permute(3,0,1,2)[None].to(device).float()/127.5-1
        codec=decoder.decode(target);copy=decoder.decode(previous)
        baselines['taesd_roundtrip'].append(measurements(target,target,codec,truth))
        baselines['prev_chunk_copy'].append(measurements(previous,target,copy,truth))
        noise=torch.randn(previous.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+i))
        for n in (1,2,4):
            with autocast_for(device):pred=model.sample_chunk(previous,tokens,n,ages,noise=noise)
            pixels=decoder.decode(pred.float());results[str(n)].append(measurements(pred,target,pixels,truth))
            if n==1 and item['source_id'] not in shown and len(shown)<4:
                shown.add(item['source_id']);visual.append((truth[0].cpu(),pixels[0].cpu(),codec[0].cpu()))
        if controls:
            ep=dataset.episodes[item['episode']];times=ep['timestamps']
            anchor_times=times[item['start']+torch.tensor([0,2,4,6])]
            frozen_ages=ages[:,0:1]+(anchor_times-anchor_times[0]).to(device=device,dtype=ages.dtype)[None]
            variants={
                'swapped_uve':(previous,other['uve_tokens'][None].to(device),other['stage_ages'][None].to(device)),
                'frozen_stage_uve':(previous,tokens[:,0:1].expand(-1,4,-1,-1),frozen_ages),
                'zero_z_prev':(torch.zeros_like(previous),tokens,ages),
                'swapped_z_prev':(other['z_prev'][None].to(device),tokens,ages)}
            for name,(prev,context,age) in variants.items():
                with autocast_for(device):pred=model.sample_chunk(prev,context,1,age,noise=noise)
                controls[name].append(measurements(pred,target,decoder.decode(pred.float()),truth))
        windows.append({'source_id':item['source_id'],'start':item['start'],'noise_seed':seed+i,
            'donor_source_id':other['source_id'],'stage_frames':item['stage_frames'].tolist(),'stage_ages':item['stage_ages'].tolist()})
        print(f'DIAGNOSIS {dataset.split} {i+1}/{len(indices)}',flush=True)
    videos=[(label,torch.cat([v[col] for v in visual],dim=1)) for col,label in enumerate(('Reference','1-NFE','Codec'))]
    contact_sheet(videos,Path(output)/f'{dataset.split}_contact_sheet.png',tuple(j for k in range(len(visual)) for j in (k*8,k*8+7)))
    report={'windows':windows,'nfe':{k:aggregate(v) for k,v in results.items()},
        'baselines':{k:aggregate(v) for k,v in baselines.items()},'conditioning_1nfe':{'baseline':aggregate(results['1']),**{k:aggregate(v) for k,v in controls.items()}}}
    atomic_json(Path(output)/f'{dataset.split}_diagnosis.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--reference-metrics');p.add_argument('--expected-step',type=int);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True);device=select_device(a.device)
    model,_,config=load_checkpoint(a.checkpoint,device)
    if a.expected_step is not None and config['step']!=a.expected_step:raise ValueError('Unexpected checkpoint step')
    decoder=PretrainedTAESD().to(device);reports={}
    for split in ('train','val'):
        dataset=CachedEpisodeChunkDataset(a.data,split,jitter=False)
        require_uve_grid(config,dataset)
        reports[split]=diagnose_split(model,decoder,dataset,device,output)
        del dataset
    reference=json.loads(Path(a.reference_metrics).read_text()) if a.reference_metrics else None
    expected=[(r['source_id'],r['start']) for r in reference['per_chunk']['1']] if reference else None
    actual=[(r['source_id'],r['start']) for r in reports['val']['windows']]
    delta=reports['val']['nfe']['1']['psnr_db']-reference['nfe']['1']['psnr_db'] if reference else None
    matched=actual==expected and abs(delta)<=.05 if reference else None
    atomic_json(output/'diagnosis.json',{'checkpoint_step':config['step'],
        'checkpoint_sha256':hashlib.sha256((Path(a.checkpoint)/'model.safetensors').read_bytes()).hexdigest(),
        'seed':917,'windows_per_split':40,'test_accessed':False,'training_performed':False,'splits':reports,
        'baseline_reproduction':{'matched':matched,'psnr_delta_db':delta,'tolerance_db':.05,'window_order_identical':actual==expected},
        'limitations':['Different-scene/zero-history controls are out of distribution; sensitivity is not proof of a unique bottleneck.',
            'Train uses one first window per scene; validation retains the exact original 40 windows. Sampling distributions differ.',
            'Frozen-stage control ages the original observation honestly; no future camera snapshots or all-zero token conditions.',
            'Codec latent L1 is zero by construction; its meaningful reference metrics are pixel PSNR/L1.']})
    if reference and not matched:raise RuntimeError('Frozen validation baseline not reproduced; do not interpret ablations yet')


if __name__=='__main__':main()
