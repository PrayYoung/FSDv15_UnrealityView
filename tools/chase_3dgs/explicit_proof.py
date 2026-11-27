"""Zero-training metric depth -> calibrated explicit splats; isolated GPU tool."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from unrealityview.data.cache import atomic_json
from tools.chase_3dgs.diagnostic_common import FRAMES,SAMPLES,SIZE,chase_pose,seal_views
from tools.chase_3dgs.native_scene import NativeScene


def timed_cuda(action):
    import torch
    torch.cuda.synchronize();wall=time.perf_counter();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();value=action();end.record();torch.cuda.synchronize()
    return value,dict(cuda_elapsed_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-wall)*1000)


def load_depth():
    import torch
    from huggingface_hub import hf_hub_download
    pins=json.loads(Path('configs/depth_metric_pins.json').read_text())
    source=Path('.cache/third_party/depth_metric/metric_depth').resolve();sys.path.insert(0,str(source))
    from depth_anything_v2.dpt import DepthAnythingV2
    p=pins['model'];path=hf_hub_download(p['repo'],p['filename'],revision=p['revision'])
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Depth model checksum mismatch')
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=80)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));return model.cuda().eval(),pins


def run(native,work,output):
    import torch
    import torch.nn.functional as F
    from unrealityview.explicit3d.project import project,rasterize,lidar_anchor,align_camera_scales
    torch.manual_seed(42);torch.set_num_threads(8)
    work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    scene=NativeScene(native);model,pins=load_depth();device='cuda';groups={};refs={};records=[];calibrations=[];orbit={}
    for source in ('camera_only','lidar_anchored'):
        for pose in ('A','B'):groups[f'{source}_{pose}']={}
    def tensor(value):return torch.as_tensor(value,device=device,dtype=torch.float32)
    def save(image,path):
        path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.rint(image.detach().cpu().clamp(0,1).numpy()*255).astype('uint8')).save(path)
    for index in FRAMES:
        frame_start=time.perf_counter();rgb,k,poses,offsets=scene.cameras(index)
        camera_load_ms=(time.perf_counter()-frame_start)*1000
        upload_start=time.perf_counter();front=poses[0];colors=tensor(rgb)/255;k=tensor(k);poses=tensor(poses)
        torch.cuda.synchronize();upload_ms=(time.perf_counter()-upload_start)*1000
        preprocessing=time.perf_counter()
        # Upstream expects BGR; all source preprocessing is pinned and frozen.
        batch=torch.cat([model.image2tensor(image[:,:,::-1].copy(),pins['input_size'])[0] for image in rgb])
        preprocess_ms=(time.perf_counter()-preprocessing)*1000
        with torch.no_grad():
            depth,timing=timed_cuda(lambda:F.interpolate(model(batch)[:,None],(SIZE[1],SIZE[0]),mode='bilinear',align_corners=True)[:,0])
            (camera_depth,cam_scales,edges),alignment_time=timed_cuda(lambda:align_camera_scales(depth,k,poses))
            lidar_start=time.perf_counter();world=tensor(scene.lidar_world(index));sparse=[]
            def sparse_views():
                zero=torch.zeros_like(world)
                for cam in range(6):sparse.append(rasterize(world,zero,poses[cam],k[cam],SIZE)[2])
                return torch.stack(sparse)
            sparse_depth,lidar_gpu_time=timed_cuda(sparse_views)
            anchored,scales,counts=lidar_anchor(depth,sparse_depth)
            lidar_total_ms=(time.perf_counter()-lidar_start)*1000
            ref=work/'reference'/f'{index:05d}.png';ref.parent.mkdir(exist_ok=True);scene.front(index,SIZE).save(ref);refs[index]=str(ref.relative_to(work))
            row=dict(index=index,timestamp=float(scene.timestamps[index]),camera_timestamp_offsets=offsets,depth=timing,camera_load_ms=camera_load_ms,input_upload_ms=upload_ms,
                depth_preprocess_ms=preprocess_ms,overlap_alignment=alignment_time,camera_only_scales=cam_scales.cpu().tolist(),overlap_edges=edges,
                lidar_scales=scales.cpu().tolist(),lidar_anchor_counts=counts,lidar_projection=lidar_gpu_time,lidar_total_wall_ms=lidar_total_ms,projections={})
            for source,predicted in (('camera_only',camera_depth),('lidar_anchored',anchored)):
                for name in ('A','B'):
                    key=f'{source}_{name}'
                    for size in (SIZE,(512,320)):
                        pose,tk=chase_pose(front,name,size);target=tensor(pose);target_k=tensor(tk)
                        (rough,mask,_),projection_time=timed_cuda(lambda:project(colors,predicted,k,poses,target,target_k,size))
                        row['projections'][f'{key}_{size[0]}x{size[1]}']=dict(hole_fraction=1-float(mask.float().mean()),**projection_time)
                        if size==SIZE:
                            path=work/key/f'{index:05d}.png';save(rough,path);groups[key][index]=str(path.relative_to(work))
                            path=work/(key+'_mask')/f'{index:05d}.png';path.parent.mkdir(exist_ok=True)
                            Image.fromarray(mask.cpu().numpy().astype('uint8')*255).save(path)
                            calibrations.append(dict(index=index,group=key,camera_to_world=pose.tolist(),intrinsics=tk.tolist(),size=list(size)))
                        elif index in SAMPLES:
                            save(rough,output/'preview_512x320'/key/f'{index:05d}.png')
                if index==55:
                    orbit[source]={}
                    for yaw in (-25,0,25):
                        pose,tk=chase_pose(front,'B',yaw=yaw)
                        rough,mask,_=project(colors,predicted,k,poses,tensor(pose),tensor(tk),SIZE)
                        path=work/'orbit'/source/f'{yaw:+03d}.png';save(rough,path);orbit[source][yaw]=str(path.relative_to(work))
            row['front_end_frame_wall_ms']=(time.perf_counter()-frame_start)*1000
            records.append(row);atomic_json(output/'projection_metrics.json',dict(frames=records,depth_model=pins,M_definition='1=valid observed projection;hole fraction=1-M.mean()',camera_only_alignment='Approximate overlap scale graph anchored to front metric scale;no LiDAR',lidar_note='Nearest synchronized world-space LiDAR per camera for median scale only;not novel-view truth or a guaranteed upper bound',timing_note='CUDA event elapsed includes any host-induced device idle;wall times also reported. No real-time claim. File export and all diagnostic variants inflate front_end_frame_wall_ms.'))
            atomic_json(output/'progress.json',dict(stage='explicit_projection',completed_frames=len(records),total_frames=68))
    atomic_json(output/'calibration.json',calibrations)
    seal_views(work,groups,refs,scene.timestamps,dict(experiment='explicit3d',depth_model=pins,orbit=orbit,source_archive=json.loads((native/'.archive.json').read_text()),training_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))
