"""One camera-only t/t-5 diagnostic with causal depth cache and GPU FP16 depth."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from unrealityview.data.cache import atomic_json,resolve_safe_subpath
from tools.chase_3dgs.diagnostic_common import FRAMES,SAMPLES,SIZE,chase_pose,comparison,preview
from tools.chase_3dgs.native_scene import NativeScene


def hole_stats(mask):
    # Same image-fraction proxy as original rows67:192; at384high use134:384.
    start=round(mask.shape[0]*67/192)
    return dict(whole=1-float(mask.float().mean()),lower_band=1-float(mask[start:].float().mean()))


def load_teacher(root):
    record=json.loads((root/'manifest.json').read_text());images={}
    for row in record['frames']:
        path=resolve_safe_subpath(root,row['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Teacher sample checksum mismatch')
        with Image.open(path) as image:images[row['index']]=image.convert('RGB')
    if set(images)!=set(SAMPLES):raise ValueError('Expected five exact poseB teacher sample frames')
    return record,images


def run(native,teacher,output):
    import torch
    import torch.nn.functional as F
    from unrealityview.explicit3d.project import project,align_camera_scales,composite_live_first
    from unrealityview.explicit3d.preprocess import preprocess_rgb
    from tools.chase_3dgs.explicit_proof import load_depth
    torch.manual_seed(42);torch.set_num_threads(8)
    output.mkdir(parents=True,exist_ok=True);teacher_record,teacher_images=load_teacher(teacher)
    scene=NativeScene(native,use_lidar=False);model,pins=load_depth();summaries={};calibrations=[]
    def tensor(value):return torch.as_tensor(value,device='cuda',dtype=torch.float32)
    def pil(value):return Image.fromarray(np.rint(value.detach().cpu().float().clamp(0,1).numpy()*255).astype('uint8'))
    def mask_image(value):return Image.fromarray(value.cpu().numpy().astype('uint8')*255).convert('RGB')
    with torch.inference_mode():
        for input_size in (518,392):
            cache={};rows=[];preroll=[];gif=[];columns=[{} for _ in range(5)]
            for index in range(7,80):
                acquire_start=time.perf_counter();raw,k_np,poses_np,offsets=scene.cameras(index)
                load_ms=(time.perf_counter()-acquire_start)*1000
                target_np,target_k_np=chase_pose(poses_np[0],'B');upload_start=time.perf_counter()
                raw_gpu=torch.as_tensor(raw,device='cuda');k=tensor(k_np);poses=tensor(poses_np)
                target=tensor(target_np);target_k=tensor(target_k_np)
                torch.cuda.synchronize();upload_ms=(time.perf_counter()-upload_start)*1000
                events={}
                def mark(name):
                    event=torch.cuda.Event(enable_timing=True);event.record();events[name]=event
                wall=time.perf_counter();mark('start')
                colors=raw_gpu.float()/255
                batch=preprocess_rgb(colors,input_size);mark('preprocess')
                with torch.autocast('cuda',dtype=torch.float16):prediction=model(batch)
                depth=F.interpolate(prediction.float()[:,None],(SIZE[1],SIZE[0]),mode='bilinear',align_corners=True)[:,0]
                mark('depth')
                if not torch.isfinite(depth).all():raise ValueError('Nonfinite FP16 depth;no automatic FP32 fallback')
                aligned,scales,edges=align_camera_scales(depth,k,poses);mark('alignment')
                cache[index]=(colors,aligned,k,poses)
                if index>=12:
                    lag=float(scene.timestamps[index]-scene.timestamps[index-5])
                    if not .44<=lag<=.56:raise ValueError('t-5 timestamp difference is not approximately0.5seconds')
                    current,mask_current,_=project(*cache[index],target,target_k,SIZE);mark('project_current')
                    past,mask_past,_=project(*cache[index-5],target,target_k,SIZE);mark('project_past')
                    combined,mask_combined=composite_live_first(current,mask_current,past,mask_past);mark('composite')
                mark('end');torch.cuda.synchronize()
                pipeline_wall_ms=(time.perf_counter()-wall)*1000
                frame_wall_ms=(time.perf_counter()-acquire_start)*1000
                def duration(a,b):return events[a].elapsed_time(events[b])
                stage=dict(preprocess_ms=duration('start','preprocess'),depth_ms=duration('preprocess','depth'),alignment_ms=duration('depth','alignment'),gpu_resident_total_ms=duration('start','end'),gpu_resident_wall_ms=pipeline_wall_ms)
                if index<12:
                    preroll.append(dict(index=index,**stage));continue
                stage.update(project_current_ms=duration('alignment','project_current'),project_past_ms=duration('project_current','project_past'),composite_ms=duration('project_past','composite'))
                # Verification/export are excluded from measured processing times.
                deviation=float((combined[mask_current]-current[mask_current]).abs().max()) if mask_current.any() else 0.
                if deviation!=0:raise ValueError('Live-priority invariant violated')
                row=dict(index=index,past_index=index-5,lag_seconds=lag,past_depth_cache_hit=True,camera_load_ms=load_ms,input_upload_ms=upload_ms,frame_wall_with_file_io_ms=frame_wall_ms,camera_timestamp_offsets=offsets,camera_scales=scales.cpu().tolist(),overlap_edges=edges,single=hole_stats(mask_current),two_timestamp=hole_stats(mask_combined),newly_filled_fraction=float((mask_combined&~mask_current).float().mean()),live_covered_max_rgb_deviation=deviation,**stage)
                rows.append(row)
                gif.append(preview(pil(combined),(320,192)))
                if index in SAMPLES:
                    images=[Image.fromarray(raw[0]),pil(current),pil(combined),mask_image(mask_current),mask_image(mask_combined)]
                    for group,image in zip(columns,images):group[index]=image
                    for name,image in zip(('single','two_timestamp','single_M','two_timestamp_M'),images[1:]):
                        path=output/'samples'/str(input_size)/name/f'{index:05d}.png';path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
                    calibrations.append(dict(index=index,input_size=input_size,target_camera_to_world=target_np.tolist(),target_intrinsics=target_k_np.tolist(),target_size=list(SIZE)))
                for key in list(cache):
                    if key<index-5:del cache[key]
                atomic_json(output/f'metrics_{input_size}.json',dict(frames=rows,preroll=preroll,teacher=teacher_record,depth_model=pins))
                atomic_json(output/'progress.json',dict(stage='two_timestamp_projection',input_size=input_size,completed_frames=len(rows),total_frames=68))
            comparison(columns+[teacher_images],['Native front','Single-frame rough','2-ts live-priority rough','Single M: white=valid','2-ts M: white=valid','PoseB OmniRe+Difix']).save(output/f'inspection_{input_size}.png')
            gif_name='explicit3d_2ts_raw.gif' if input_size==518 else 'explicit3d_2ts_raw_392.gif'
            if gif:gif[0].save(output/gif_name,save_all=True,append_images=gif[1:],duration=1000*float(np.median(np.diff(scene.timestamps[12:80]))),loop=0)
            def stats(values):
                return dict(mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.quantile(values,.95)))
            summaries[str(input_size)]=dict(single_holes={band:stats([r['single'][band] for r in rows]) for band in ('whole','lower_band')},two_timestamp_holes={band:stats([r['two_timestamp'][band] for r in rows]) for band in ('whole','lower_band')},gpu_resident_total_ms=stats([r['gpu_resident_total_ms'] for r in rows]),gpu_resident_fps_from_median=1000/float(np.median([r['gpu_resident_total_ms'] for r in rows])),frame_wall_with_file_io_ms=stats([r['frame_wall_with_file_io_ms'] for r in rows]),wall_fps_with_file_io_from_median=1000/float(np.median([r['frame_wall_with_file_io_ms'] for r in rows])),stage_ms={key:stats([r[key] for r in rows]) for key in ('preprocess_ms','depth_ms','alignment_ms','project_current_ms','project_past_ms','composite_ms','camera_load_ms','input_upload_ms')},live_priority_verified=all(r['live_covered_max_rgb_deviation']==0 for r in rows),preroll_frames=5,cache_policy='Steady state one6-camera depth inference per frame;reuse t-5 cached realcamera depth,not previous generatedlatent',ghosting_guarantee='Current-covered pixels unchanged only;historical content in current holes may be stale')
            atomic_json(output/'summary.json',dict(status='completed_variant',variants=summaries,output_size=list(SIZE),frames=FRAMES,lidar_used=False,z_prev_used=False,difix_used=False,precision='autocast_cuda_float16_for_depth;float32geometry',gpu_timing='Device-resident preprocessing+depth+alignment+two projections+composite;includes host launch/solver gaps;excludes file I/O,host upload,model load,preroll and diagnostic export',lower_band_rows_at384=[134,384],equivalent_lower_band_rows_at192=[67,192],gif_variant=518,visual_approval=False,step2_launched=False,step3_launched=False))
    atomic_json(output/'calibration.json',calibrations)
    final=json.loads((output/'summary.json').read_text());final['status']='completed';atomic_json(output/'summary.json',final)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(**vars(p.parse_args()))
