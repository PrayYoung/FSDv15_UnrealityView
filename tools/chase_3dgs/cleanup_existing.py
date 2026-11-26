"""Standalone Difix on exact existing RGBs; no reconstruction or native archive."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from unrealityview.data.cache import atomic_json,resolve_safe_subpath
from tools.chase_3dgs.contract import chase_calibration,project_world
from unrealityview.data.real_world import CV_TO_FLU


def fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    calibration=resolve_safe_subpath(root,manifest['calibration'])
    if fingerprint(calibration)!=manifest['calibration_sha256']:raise ValueError('Calibration fingerprint mismatch')
    cameras=json.loads(calibration.read_text())
    if len(cameras)!=80:raise ValueError('Expected80 calibration frames')
    front=np.eye(4);front[:3,:3]=CV_TO_FLU
    _,expected_k=chase_calibration(front)
    for camera in cameras:
        if np.asarray(camera['camera_to_world']).shape!=(4,4) or not np.isfinite(camera['camera_to_world']).all():raise ValueError('Invalid calibration pose')
        np.testing.assert_allclose(camera['intrinsics'],expected_k,atol=1e-6)
    frames=manifest['frames']
    if manifest['sequence']!='040' or len(frames)!=80 or [f['index'] for f in frames]!=list(range(80)):
        raise ValueError('Standalone test requires all80 ordered scene040 frames')
    timestamps=[float(f['timestamp']) for f in frames]
    if not np.isfinite(timestamps).all() or (np.diff(timestamps)<=0).any():raise ValueError('Invalid native timestamps')
    raw=[];references=[]
    for row in frames:
        for field,destination in (('raw',raw),('reference',references)):
            path=resolve_safe_subpath(root,row[field])
            if fingerprint(path)!=row[field+'_sha256']:raise ValueError('Input fingerprint mismatch')
            with Image.open(path) as image:
                if image.size!=(320,192):raise ValueError('Input image dimensions changed')
                destination.append(image.convert('RGB'))
    return manifest,raw,references,timestamps


def marker_pixels():
    front=np.eye(4);front[:3,:3]=CV_TO_FLU
    pose,k=chase_calibration(front)
    return project_world(np.array([[15,0,0],[25,0,0],[40,0,0]]),pose,k)


def inspection(raw,references,cleaned):
    result=Image.new('RGB',(960,660),'#111820');draw=ImageDraw.Draw(result);uv=marker_pixels()
    for column,index in enumerate((0,39,79)):
        for row,(name,source) in enumerate(zip(('Native front reference','Raw3DGS + overlay markers','Difix + identical overlay markers'),(references[index],raw[index],cleaned[index]))):
            image=source.copy()
            if row:
                painter=ImageDraw.Draw(image)
                for x,y in uv:painter.ellipse((x-3,y-3,x+3,y+3),fill='#258cff')
            result.paste(image,(column*320,row*220+28));draw.text((column*320+5,row*220+5),f'{name} / {index}',fill='white')
    return result


def diagnostic_metrics(raw,cleaned):
    raw=np.stack([np.asarray(i).astype('float32')/255 for i in raw]);clean=np.stack([np.asarray(i).astype('float32')/255 for i in cleaned])
    def artifacts(x):
        lower=x[:,67:];r,g,b=np.moveaxis(lower,-1,0)
        return dict(cyan_fraction_lower_band=float(((b>.45)&(g>.35)&(b-r>.2)&(g-r>.05)).mean()),
                    purple_fraction_lower_band=float(((r>.25)&(b>.25)&(g<np.minimum(r,b)*.8)).mean()))
    a=raw[:,67:].mean(-1);b=clean[:,67:].mean(-1)
    ay,ax=np.gradient(a,axis=(1,2));by,bx=np.gradient(b,axis=(1,2))
    cosine=float(((ax*bx+ay*by).sum())/max(float(np.sqrt((ax*ax+ay*ay).sum()*(bx*bx+by*by).sum())),1e-12))
    return dict(raw_color_proxies=artifacts(raw),clean_color_proxies=artifacts(clean),
        raw_frame_delta_l1=float(np.abs(np.diff(raw,axis=0)).mean()),clean_frame_delta_l1=float(np.abs(np.diff(clean,axis=0)).mean()),
        edit_residual_frame_delta_l1=float(np.abs(np.diff(clean-raw,axis=0)).mean()),
        raw_clean_pixel_l1=float(np.abs(clean-raw).mean()),lower_band_gradient_cosine=cosine,
        note='Color and edge proxies are descriptive,not geometry/flicker truth;native front reference is a different view. No novel-view PSNR or new alpha is claimed.')


def run_cleanup(inputs,output,cleaner,pins):
    start=time.monotonic();manifest,raw,references,timestamps=load_inputs(inputs);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    timings={'input_seconds':time.monotonic()-start};events={}
    def progress(stage,details):
        if stage=='model_loaded':events['model_load_seconds']=details['seconds']
        if stage!='frame' or details['completed_frames'] in (1,10,20,40,60,80):
            atomic_json(output/'progress.json',dict(stage=stage,**details))
    cleanup_start=time.monotonic();cleaned=cleaner(raw,references,pins,progress=progress)
    if len(cleaned)!=80 or any(frame.size!=(320,192) for frame in cleaned):raise ValueError('Difix frame count/geometry changed')
    timings['cleanup_including_model_seconds']=time.monotonic()-cleanup_start;timings.update(events)
    timings['inference_seconds']=timings['cleanup_including_model_seconds']-timings.get('model_load_seconds',0)
    export_start=time.monotonic();directory=output/'rgb_clean';directory.mkdir(exist_ok=True);frames=[]
    for index,frame in enumerate(cleaned):
        path=directory/f'{index:05d}.png';frame.convert('RGB').save(path);frames.append({'index':index,'sha256':fingerprint(path)})
    inspection(raw,references,cleaned).save(output/'inspection.png')
    cleaned[0].save(output/'chase_difix.gif',save_all=True,append_images=cleaned[1:],duration=1000*float(np.median(np.diff(timestamps))),loop=0)
    atomic_json(output/'metrics.json',diagnostic_metrics(raw,cleaned));timings['export_seconds']=time.monotonic()-export_start;timings['total_seconds']=time.monotonic()-start
    atomic_json(output/'timings.json',timings)
    atomic_json(output/'run_receipt.json',dict(status='completed',sequence='040',frames=80,difix_cleanup=True,mode='standalone_existing_raw_frames',
        raw_source_run=manifest['raw_source_run'],input_manifest_sha256=fingerprint(Path(inputs)/'manifest.json'),
        calibration_sha256=manifest['calibration_sha256'],model=pins['difix_model'],code=pins['difix'],
        seed=42,raw_refit=False,archive_downloaded=False,output_frames=frames,
        markers='Blue calibration dots are posthoc identical inspection overlays;not present in model input/output RGB. Fixed overlays do not prove correct generated geometry.',
        alpha='Raw opacity statistics unchanged;Difix changes only RGB',visual_review_required=True))
    return timings


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.manual_seed(42)
    from tools.chase_3dgs.render_chase_teacher import cleanup_difix
    pins=json.loads(Path('tools/chase_3dgs/upstream_pins.json').read_text())
    run_cleanup(a.inputs,a.output,cleanup_difix,pins)

if __name__=='__main__':main()
