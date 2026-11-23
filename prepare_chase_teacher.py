"""Experimental camera-only static 3DGS teacher; per-scene fitting, no LiDAR.

Uses calibrated SIFT triangulation to initialize gsplat. Dynamic objects and unseen
surfaces are not reconstructed reliably. Native timestamps are preserved. Quality
and alpha-coverage gates must pass before an importer-readable manifest is written.
"""
import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file
from unrealityview.data.cache import atomic_json,resolve_safe_subpath
from unrealityview.data.real_world import PandaSetSequence,pose_matrix,CV_TO_FLU


def chase_calibration(front_pose,width=320,height=192,back=6.8,height_m=3.8,pitch=19.,fov=75.):
    """World-from-OpenCV chase pose relative to front-center FLU reference."""
    ego=front_pose.copy();ego[:3,:3]=front_pose[:3,:3]@CV_TO_FLU.T
    angle=math.radians(pitch);c,s=math.cos(angle),math.sin(angle)
    relative=np.eye(4)
    relative[:3,:3]=np.array([[0,-s,c],[-1,0,0],[0,-c,-s]])
    relative[:3,3]=[-back,0,height_m]
    focal=width/(2*math.tan(math.radians(fov)/2))
    return ego@relative,np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]])


def triangulate_matches(points1,points2,k1,k2,pose1,pose2,max_error=1.5):
    import cv2
    p1=k1@np.linalg.inv(pose1)[:3];p2=k2@np.linalg.inv(pose2)[:3]
    homogeneous=cv2.triangulatePoints(p1,p2,points1.T,points2.T).T
    valid=abs(homogeneous[:,3])>1e-8
    world=homogeneous[:,:3]/np.where(valid,homogeneous[:,3],1)[:,None]
    xyz1=(np.linalg.inv(pose1)@np.c_[world,np.ones(len(world))].T).T
    xyz2=(np.linalg.inv(pose2)@np.c_[world,np.ones(len(world))].T).T
    projection=lambda p:(p[:,:2]/np.maximum(p[:,2:3],1e-8))
    repro1=projection((k1@xyz1[:,:3].T).T);repro2=projection((k2@xyz2[:,:3].T).T)
    # Reject near-parallel rays as well as negative depth and reprojection errors.
    ray1=world-pose1[:3,3];ray2=world-pose2[:3,3]
    cosine=(ray1*ray2).sum(1)/(np.linalg.norm(ray1,axis=1)*np.linalg.norm(ray2,axis=1)+1e-8)
    valid&=(xyz1[:,2]>.5)&(xyz2[:,2]>.5)&(xyz1[:,2]<120)&(cosine<math.cos(math.radians(.5)))
    valid&=(np.linalg.norm(repro1-points1,axis=1)<max_error)&(np.linalg.norm(repro2-points2,axis=1)<max_error)
    valid&=np.isfinite(world).all(1)
    return world[valid].astype(np.float32),valid


def initialize_points(views,max_points=60000):
    import cv2
    sift=cv2.SIFT_create(nfeatures=1500);features=[];cloud=[];colors=[]
    for view in views:
        key,desc=sift.detectAndCompute(cv2.cvtColor(view['rgb'],cv2.COLOR_RGB2GRAY),None)
        features.append((np.array([k.pt for k in key],np.float32),desc))
    for index,view in enumerate(views):
        # Views are interleaved by six cameras. Same camera at the next sampled
        # time and adjacent camera at this time provide calibrated baselines.
        for other in (index+6,index+1 if index%6<5 else len(views)):
            if other>=len(views):continue
            xy1,d1=features[index];xy2,d2=features[other]
            if d1 is None or d2 is None or min(len(d1),len(d2))<2:continue
            matches=cv2.BFMatcher().knnMatch(d1,d2,k=2)
            good=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
            if len(good)<8:continue
            points1=np.array([xy1[m.queryIdx] for m in good]);points2=np.array([xy2[m.trainIdx] for m in good])
            xyz,valid=triangulate_matches(points1,points2,view['k'],views[other]['k'],view['pose'],views[other]['pose'])
            pixels=np.rint(points1[valid]).astype(int);pixels[:,0]=pixels[:,0].clip(0,223);pixels[:,1]=pixels[:,1].clip(0,139)
            cloud.append(xyz);colors.append(view['rgb'][pixels[:,1],pixels[:,0]]/255.)
    if not cloud or sum(map(len,cloud))<100:raise ValueError('Insufficient calibrated feature triangulation; teacher fit aborted')
    xyz=np.concatenate(cloud);rgb=np.concatenate(colors).astype(np.float32)
    _,unique=np.unique(np.rint(xyz/.05).astype(np.int64),axis=0,return_index=True)
    indices=unique[np.random.default_rng(42).permutation(len(unique))[:max_points]]
    return xyz[indices],rgb[indices]


def fit(args):
    if not torch.cuda.is_available():raise RuntimeError('gsplat teacher requires CUDA; use a managed remote GPU job')
    from gsplat import rasterization
    sequence=PandaSetSequence(args.pandaset_root,args.sequence);views=[];held_out=[]
    # Each invocation fits ONE sequence; no shared weights or point cloud across splits.
    for index in range(0,len(sequence),args.frame_stride):
        frame=sequence.frame(index)
        for camera,(image,k,stamp) in enumerate(zip(frame.images,frame.intrinsics,frame.camera_timestamps)):
            native=sequence.cameras[camera];nearest=int(np.abs(native[2]-stamp).argmin())
            view=dict(rgb=np.array(image),k=k,pose=pose_matrix(native[1][nearest]))
            (held_out if index%args.holdout_every==0 else views).append(view)
    if len(views)<12 or not held_out:raise ValueError('Need enough training and held-out camera views')
    xyz,rgb=initialize_points(views,args.max_points)
    # Center world coordinates for stable float32 rasterization without exposing location.
    origin=np.median(xyz,axis=0);xyz-=origin
    for view in views+held_out:view['pose'][:3,3]-=origin
    device='cuda';torch.manual_seed(args.seed)
    parameters=torch.nn.ParameterDict({
        'means':torch.nn.Parameter(torch.tensor(xyz,device=device)),
        'quats':torch.nn.Parameter(torch.tensor([[1.,0,0,0]],device=device).repeat(len(xyz),1)),
        'log_scales':torch.nn.Parameter(torch.full((len(xyz),3),math.log(.12),device=device)),
        'opacity':torch.nn.Parameter(torch.zeros(len(xyz),device=device)),
        'color':torch.nn.Parameter(torch.logit(torch.tensor(rgb,device=device).clamp(.01,.99)))})
    optimizer=torch.optim.Adam([dict(params=[p],lr=.001 if k=='means' else .01) for k,p in parameters.items()])
    def render(pose,k,width=224,height=140):
        return rasterization(means=parameters['means'],quats=torch.nn.functional.normalize(parameters['quats'],dim=-1),
            scales=parameters['log_scales'].clamp(-6,2).exp(),opacities=parameters['opacity'].sigmoid(),
            colors=parameters['color'].sigmoid(),viewmats=torch.tensor(np.linalg.inv(pose),device=device,dtype=torch.float32)[None],
            Ks=torch.tensor(k,device=device,dtype=torch.float32)[None],width=width,height=height,packed=False)
    generator=np.random.default_rng(args.seed)
    for step in range(args.steps):
        view=views[int(generator.integers(len(views)))];optimizer.zero_grad(set_to_none=True)
        pixels,alpha,_=render(view['pose'],view['k'])
        truth=torch.tensor(view['rgb'],device=device,dtype=torch.float32)[None]/255
        loss=(pixels-truth).abs().mean()+.001*parameters['log_scales'].exp().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite teacher loss')
        loss.backward();optimizer.step()
        if step%100==0:print(f'TEACHER step={step} loss={float(loss):.5f}',flush=True)
    output=resolve_safe_subpath(args.output,f'pandaset/{args.sequence}');output.mkdir(parents=True,exist_ok=True)
    # Invalidate any prior accepted manifest until this run passes both gates.
    (output/'manifest.json').unlink(missing_ok=True)
    with torch.no_grad():
        psnrs=[]
        for view in held_out:
            pixels,_,_=render(view['pose'],view['k']);truth=torch.tensor(view['rgb'],device=device)/255
            psnrs.append(float(-10*torch.log10((pixels[0]-truth).square().mean().clamp_min(1e-12))))
        frames=[];coverage=[]
        for index in range(len(sequence)):
            pose,k=chase_calibration(pose_matrix(sequence.cameras[1][1][index]));pose[:3,3]-=origin
            pixels,alpha,_=render(pose,k,320,192)
            coverage.append(float((alpha>.5).float().mean()))
            relative=f'rgb/{index:05d}.png';(output/'rgb').mkdir(exist_ok=True)
            Image.fromarray((pixels[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(output/relative);frames.append(relative)
        report=dict(held_out_psnr_db=float(np.mean(psnrs)),chase_coverage_min=min(coverage),
                    points=len(xyz),steps=args.steps,source_id=f'pandaset/{args.sequence}',
                    limitation='Static 3DGS; moving objects, occlusion holes and unseen surfaces remain limitations',
                    interpolation='none; native timestamps',accepted=False)
        report['accepted']=report['held_out_psnr_db']>=args.min_psnr and report['chase_coverage_min']>=args.min_coverage
        atomic_json(output/'quality.json',report)
        save_file({k:v.detach().cpu().contiguous() for k,v in parameters.items()},str(output/'teacher.safetensors'))
        if not report['accepted']:raise RuntimeError('Teacher quality gates failed; no importer manifest published. Inspect quality.json and RGBs')
        atomic_json(output/'manifest.json',dict(source_id=f'pandaset/{args.sequence}',target_view='elevated_chase',
            teacher='Experimental calibrated SIFT + static gsplat 3DGS v1',
            camera=dict(camera_back=6.8,camera_height=3.8,pitch_deg=19.,fov=75.),
            timestamps=sequence.timestamps.tolist(),frames=frames,quality=report))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pandaset-root',required=True);p.add_argument('--sequence',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--frame-stride',type=int,default=2)
    p.add_argument('--holdout-every',type=int,default=10);p.add_argument('--max-points',type=int,default=60000)
    p.add_argument('--min-psnr',type=float,default=18.);p.add_argument('--min-coverage',type=float,default=.7)
    p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    if min(args.steps,args.frame_stride,args.max_points)<1 or args.holdout_every<2 or not 0<=args.min_coverage<=1:p.error('Invalid teacher bounds')
    fit(args)


if __name__=='__main__':main()
