"""Local, read-only 4x8 parity probes. Never train, re-encode a cache or rent GPUs."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from unrealityview.data.cache import load_episode,episode_grid,atomic_json
from unrealityview.geometry import plucker_rays
from unrealityview.models.mini_uve import MiniUVE
from scripts.reencode_uve_cache import SLOTS,digest


def tensor_sha256(tensor):
    array=tensor.detach().cpu().contiguous()
    return hashlib.sha256(array.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def fingerprints(encoder):
    weights=[(k,v) for k,v in encoder.backbone.named_parameters() if k.endswith('weight')]
    if not weights:weights=list(encoder.backbone.named_parameters())
    if not weights:raise ValueError('Backbone has no weights to fingerprint')
    full=hashlib.sha256()
    for k,v in encoder.backbone.state_dict().items():
        full.update(k.encode());full.update(str(tuple(v.shape)).encode());full.update(str(v.dtype).encode());full.update(bytes.fromhex(tensor_sha256(v)))
    def describe(pair):
        name,value=pair
        return dict(name=name,shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_sha256(value))
    return dict(ray_projection=describe(('ray_projection',encoder.ray_projection)),
        backbone_first_weight=describe(weights[0]),backbone_last_weight=describe(weights[-1]),
        backbone_state_sha256=full.hexdigest(),
        note='Current encoder fingerprints; historical cache did not record original encoder hashes, so identity cannot be confirmed retrospectively.')


def per_camera(reference,prediction,mask):
    if reference.shape!=prediction.shape or reference.ndim!=3 or reference.shape[1:]!=(192,384):raise ValueError('Expected matching 4x8 camera tokens')
    if mask.shape!=(len(reference),6):raise ValueError('Camera mask shape mismatch')
    if not torch.isfinite(reference).all() or not torch.isfinite(prediction).all():raise ValueError('Nonfinite tokens')
    a=reference.float().reshape(len(reference),6,32,384)
    b=prediction.float().reshape_as(a);result=[]
    for i,slot in enumerate(SLOTS):
        valid=mask[:,i].bool()
        if not valid.any():result.append(dict(slot=slot,available_frames=0));continue
        x=a[valid,i].flatten().double();y=b[valid,i].flatten().double()
        rms=x.square().mean().sqrt();error=(x-y).square().mean().sqrt()
        result.append(dict(slot=slot,available_frames=int(valid.sum()),reference_rms=float(rms),
            prediction_rms=float(y.square().mean().sqrt()),error_rms=float(error),
            relative_rms=float(error/rms.clamp_min(1e-12)),
            cosine=float(torch.nn.functional.cosine_similarity(x[None],y[None],eps=1e-12))))
    return result


def read_cameras(directory,ids):
    frames=[]
    for i in ids:
        slots=[]
        for name in SLOTS:
            with Image.open(Path(directory)/'raw'/name/f'{i:05d}.jpg') as im:
                if im.size!=(224,140):raise ValueError('Expected calibrated 224x140 camera JPEGs')
                slots.append(np.array(im.convert('RGB')))
        frames.append(slots)
    return torch.from_numpy(np.array(frames)).permute(0,1,4,2,3).float()/255


@contextmanager
def precision_context(device,precision):
    if precision=='fp32':
        yield
        return
    with torch.autocast(device.type,dtype={'fp16':torch.float16,'bf16':torch.bfloat16}[precision]):
        if not torch.is_autocast_enabled(device.type):raise NotImplementedError('Requested autocast precision was disabled by backend')
        yield


@torch.no_grad()
def encode_mode(encoder,cameras,episode,ids,device,precision,batch_size,fp32_rays=False):
    semantic=[];projected=[];full=[]
    for start in range(0,len(ids),batch_size):
        selected=ids[start:start+batch_size]
        image=cameras[start:start+batch_size].to(device)
        k=episode['intrinsics'][selected].to(device);ext=episode['camera_to_ego'][selected].to(device)
        mask=episode['camera_mask'][selected].to(device)
        with precision_context(device,precision):
            sem=encoder(image,camera_mask=mask)
            if fp32_rays:
                with torch.autocast(device.type,enabled=False):rays=plucker_rays(k.float(),ext.float()).to(sem)
            else:rays=plucker_rays(k.float(),ext.float()).to(sem)
            ray=(rays@encoder.ray_projection.to(sem))*mask[:,:,None,None].to(sem.dtype)
            ray=ray.reshape_as(sem)
            combined=sem+ray
            # Verify decomposition against the unmodified MiniUVE forward on
            # the first batch of every mode. This is diagnostic, not a new encoder.
            if start==0 and (not fp32_rays or precision=="fp32"):
                actual=encoder(image,k,ext,mask)
                if not torch.equal(actual,combined):raise ValueError('Semantic/ray decomposition differs from MiniUVE')
        semantic.append(sem.float().cpu());projected.append(ray.float().cpu())
        full.append(combined.half().float().cpu()) # match the stored FP16 cache
    return dict(semantic=torch.cat(semantic),projected=torch.cat(projected),full=torch.cat(full))


def diagnose(path,encoder,device,frame_count=8,precisions=('fp32','fp16','bf16'),batch_sizes=(1,8),fp32_rays=False):
    path=Path(path);before=digest(path);episode=load_episode(path)
    meta=json.loads(path.with_name('metadata.json').read_text())
    if episode_grid(episode,meta)!=(4,8) or meta.get('split')=='test':raise ValueError('Use one non-test legacy4x8 episode')
    n=len(episode['uve_tokens']);count=n if frame_count==0 else min(n,frame_count)
    if count<1:raise ValueError('Need at least one frame')
    ids=np.linspace(0,n-1,count,dtype=int).tolist();mask=episode['camera_mask'][ids].cpu()
    fingerprint=fingerprints(encoder)
    rays=plucker_rays(episode['intrinsics'][ids].float(),episode['camera_to_ego'][ids].float())
    reference_projection=(rays@encoder.ray_projection.detach().cpu().float())*mask[:,:,None,None]
    reference_projection=reference_projection.reshape(count,192,384)
    cached=episode['uve_tokens'][ids].float().cpu()
    semantic_proxy=cached-reference_projection
    cameras=read_cameras(path.parent,ids);encoder=encoder.to(device).eval();modes={};predictions={}
    for precision in precisions:
        for batch in batch_sizes:
            name=f'{device.type}_{precision}_batch{batch}';start=time.monotonic()
            try:
                encoded=encode_mode(encoder,cameras,episode,ids,device,precision,batch,fp32_rays=fp32_rays)
                record=dict(status='completed',seconds=time.monotonic()-start,
                    cached_vs_with_rays=per_camera(cached,encoded['full'],mask),
                    cached_vs_without_rays=per_camera(cached,encoded['semantic'].half().float(),mask),
                    ray_subtracted_cached_vs_semantic=per_camera(semantic_proxy,encoded['semantic'],mask),
                    fp32_rays_vs_mode_rays=per_camera(reference_projection,encoded['projected'],mask))
                predictions[name]=encoded['full']
                # Original thresholds are only reported, never changed here.
                record['frozen_parity_limits_pass']=all(x.get('available_frames',0)==0 or (x['relative_rms']<=.20 and x['cosine']>=.98) for x in record['cached_vs_with_rays'])
                modes[name]=record
            except (RuntimeError,TypeError,NotImplementedError) as error:
                reason='CPU ray-construction dtype conflict' if 'expected scalar type Float but found' in str(error) else ('Backend disabled requested autocast precision' if isinstance(error,NotImplementedError) else 'See error type; no success assumed')
                modes[name]=dict(status='unavailable_or_failed',error_type=type(error).__name__,reason=reason,seconds=time.monotonic()-start,
                    note='Mode could not execute on this backend; no precision/backend fallback or pass assumed.')
            print(name,modes[name]['status'],flush=True)
    pairwise={}
    names=list(predictions)
    for i,a in enumerate(names):
        for b in names[i+1:]:pairwise[f'{a}__vs__{b}']=per_camera(predictions[a],predictions[b],mask)
    after=digest(path)
    if before!=after:raise RuntimeError('Source episode changed during read-only diagnostic')
    return dict(source_id=meta.get('source_id'),source_episode_sha256=before,source_unchanged=True,
        sampled_frame_ids=ids,episode_frames=n,sampled_frames=count,grid=[4,8],camera_slots=list(SLOTS),
        fingerprints=fingerprint,torch_version=torch.__version__,device=device.type,modes=modes,pairwise=pairwise,
        ray_policy='controlled_FP32_ray_construction' if fp32_rays else 'native_MiniUVE_autocast',
        projected_ray_rms_by_camera=[float(reference_projection.reshape(count,6,32,384)[:,i].square().mean().sqrt()) for i in range(6)],
        frozen_thresholds=dict(relative_rms_max=.20,cosine_min=.98),paid_compute_used=False,
        limitations=['Ray-subtracted cached tokens are a proxy: original semantic features and original mixed-precision ray rounding were not stored separately.',
            'Same JPEG inputs isolate current batching/precision differences, not JPEG damage versus unavailable original decoded pixels.',
            'Current weight hashes cannot establish historical encoder identity without historical hashes.',
            'A small evenly spaced frame sample is not identical to the prior all80-frame CUDA comparison. Unsupported modes are reported, not silently emulated.',
            'Controlled FP32 ray probes, if requested, are diagnostic-only and do not assert equivalence to native low-precision CPU MiniUVE.',
            'No cache, model parameters, parity thresholds or training settings were modified.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=8);p.add_argument('--fp32-rays',action='store_true');p.add_argument('--device',choices=('auto','mps','cpu'),default='auto');a=p.parse_args()
    target=Path(a.output).resolve();source=Path(a.episode).resolve().parent
    if target.suffix!='.json' or source in target.parents:p.error('Write a JSON report outside the source episode directory')
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu') if a.device=='auto' else torch.device(a.device)
    result=diagnose(a.episode,MiniUVE(),device,a.frames,fp32_rays=a.fp32_rays)
    atomic_json(a.output,result)


if __name__=='__main__':main()
