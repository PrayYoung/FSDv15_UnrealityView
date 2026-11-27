"""Pinned Difix at640x384; small allowlisted diagnostics for either experiment."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from unrealityview.data.cache import atomic_json
from tools.chase_3dgs.diagnostic_common import FRAMES,SAMPLES,SIZE,load_pil,preview,comparison,delta


def run(work,output):
    import torch
    torch.manual_seed(42);output.mkdir(parents=True,exist_ok=True)
    views=json.loads((work/'views.json').read_text());assert views['frames']==FRAMES and views['size']==list(SIZE)
    references={int(k):load_pil(work/v) for k,v in views['references'].items()}
    raw={g:{int(k):load_pil(work/v) for k,v in frames.items()} for g,frames in views['groups'].items()}
    # One shared model load, one frame per inference; preserve native640x384.
    batches=[];refs=[];keys=[]
    for group,images in raw.items():
        for index in FRAMES:batches.append(images[index]);refs.append(references[index]);keys.append((group,index))
    orbit=views['extra'].get('orbit',{})
    for source,angles in orbit.items():
        for yaw,path in angles.items():batches.append(load_pil(work/path));refs.append(references[55]);keys.append((f'orbit_{source}',int(yaw)))
    if any(i.size!=SIZE for i in batches+refs):raise ValueError('Difix must receive native640x384 inputs/references')
    from tools.chase_3dgs.render_chase_teacher import cleanup_difix
    pins=json.loads(Path('tools/chase_3dgs/upstream_pins.json').read_text());events={};cuda_times=[];start=time.monotonic()
    def progress(stage,details):
        if stage=='model_loaded':events['model_load_seconds']=details['seconds']
        if 'cuda_elapsed_ms' in details:cuda_times.append(details['cuda_elapsed_ms'])
        if stage!='frame' or details['completed_frames']%10==0:
            atomic_json(output/'progress.json',dict(stage='difix_'+stage,total_frames=len(keys),**details))
    cleaned=cleanup_difix(batches,refs,pins,progress=progress,cuda_timing=True);elapsed=time.monotonic()-start
    if len(cleaned)!=len(keys):raise ValueError('Cleaned frame count mismatch')
    result={}
    for (group,index),image in zip(keys,cleaned):result.setdefault(group,{})[index]=image
    metrics={}
    for group,images in raw.items():
        clean=result[group];edit=float(np.mean([np.abs(np.asarray(images[i]).astype('float32')-np.asarray(clean[i]).astype('float32')).mean()/255 for i in FRAMES]))
        metrics[group]=dict(raw_frame_delta_l1=delta(images),clean_frame_delta_l1=delta(clean),raw_clean_pixel_l1=edit)
        # Keep only five paired candidates at each preview size, not all variants.
        for i in SAMPLES:
            for label,source in (('raw',images),('difix',clean)):
                for size in ((320,192),(512,320)):
                    p=output/'samples'/f'{group}_{label}_{size[0]}x{size[1]}'/f'{i:05d}.png';p.parent.mkdir(parents=True,exist_ok=True);preview(source[i],size).save(p)
                if i in (25,40):
                    p=output/'native_samples'/f'{group}_{label}'/f'{i:05d}.png';p.parent.mkdir(parents=True,exist_ok=True);source[i].save(p)
    is_refit=views['extra']['experiment']=='refit'
    if is_refit:
        columns=[references,raw['pose_A'],result['pose_A'],raw['pose_B'],result['pose_B']]
        labels=['Native front','A raw','A Difix640x384','B raw','B Difix640x384'];main=result['pose_B'];gif='chase_poseB_difix.gif'
    else:
        masks={i:load_pil(work/'camera_only_B_mask'/f'{i:05d}.png') for i in FRAMES}
        columns=[references,raw['camera_only_B'],result['camera_only_B'],raw['lidar_anchored_B'],result['lidar_anchored_B'],masks]
        labels=['Native front','Cam-only rough B','Cam-only Difix B','LiDAR rough B','LiDAR Difix B','M: white=valid B'];main=result['camera_only_B'];gif='explicit3d_chase.gif'
        comparison([references,raw['camera_only_A'],result['camera_only_A'],raw['lidar_anchored_A'],result['lidar_anchored_A']],['Native front','Cam-only rough A','Cam-only Difix A','LiDAR rough A','LiDAR Difix A']).save(output/'inspection_poseA.png')
        for source,angles in orbit.items():
            for yaw,path in angles.items():
                for label,image in (('raw',load_pil(work/path)),('difix',result[f'orbit_{source}'][int(yaw)])):
                    p=output/'orbit'/f'{source}_{int(yaw):+03d}_{label}.png';p.parent.mkdir(exist_ok=True);preview(image,(320,192)).save(p)
        html='''<!doctype html><meta charset="utf-8"><title>Scene040 discrete orbit check</title><p>Three sampled orbit views, not continuous3D. Frame55, poseB; drag the image or use slider.</p><select id="source"><option>camera_only</option><option>lidar_anchored</option></select><select id="kind"><option>difix</option><option>raw</option></select><input id="angle" type="range" min="0" max="2" value="1" step="1"><span id="label"></span><br><img id="view" width="640" draggable="false"><script>const a=document.getElementById('angle'),s=document.getElementById('source'),k=document.getElementById('kind'),v=document.getElementById('view');function update(){const y=['-25','+00','+25'][+a.value];v.src='orbit/'+s.value+'_'+y+'_'+k.value+'.png';document.getElementById('label').textContent=y+' degrees'}[a,s,k].forEach(x=>x.oninput=update);let start;v.onpointerdown=e=>{start=e.clientX;v.setPointerCapture(e.pointerId)};v.onpointermove=e=>{if(start!==undefined){a.value=Math.max(0,Math.min(2,1+Math.round((e.clientX-start)/80)));update()}};v.onpointerup=()=>start=undefined;update()</script>'''
        (output/'orbit.html').write_text(html)
    comparison(columns,labels).save(output/'inspection.png')
    frames=[preview(main[i],(320,192)) for i in FRAMES]
    frames[0].save(output/gif,save_all=True,append_images=frames[1:],duration=1000*float(np.median(np.diff(views['timestamps']))),loop=0)
    atomic_json(output/'cleanup_metrics.json',dict(groups=metrics,frames=FRAMES,size=list(SIZE),model=pins['difix_model'],code=pins['difix'],timestep=199,guidance=0,seed=42,cleanup_seconds=elapsed,**events,inference_seconds=elapsed-events.get('model_load_seconds',0),inference_ms_per_frame=1000*(elapsed-events.get('model_load_seconds',0))/len(keys),cuda_frame_ms=cuda_times,inference_timing='Whole per-frame pipe wall latency including VAE decode and PIL transfer;CUDA events also reported,including host-induced GPU idle',mask_semantics='M=valid observed coverage;Difix receives rough RGB/reference only,not M',native_resolution_references=True,preview512_policy='Letterboxed,not stretched',frames_count=68,extra=views['extra'],visual_approval=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(**vars(p.parse_args()))
