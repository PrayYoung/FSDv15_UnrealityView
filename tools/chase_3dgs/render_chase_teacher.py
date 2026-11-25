"""Fit the pinned external dynamic scene model and export calibrated PNG/JSON.

Run ONLY in .drivestudio-venv. Native pickle tables and external .pth files stay
in isolated scratch; the main application receives RGB PNGs and JSON only.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
from tools.chase_3dgs.render_policy import alpha_coverages,consistent_novel_appearance,GATE_POLICY
from tools.chase_3dgs.contract import chase_calibration,processed_front_pose,project_world,export_manifest

ROOT=Path(__file__).resolve().parents[2]


def build_config(upstream,processed,sequence,scratch,iters):
    from omegaconf import OmegaConf
    cfg=OmegaConf.merge(OmegaConf.load(upstream/'configs/omnire.yaml'),
        OmegaConf.load(upstream/'configs/datasets/pandaset/6cams.yaml'),
        OmegaConf.load(ROOT/'tools/chase_3dgs/omnire_pandaset_6cams_nosmpl.yaml'))
    del cfg['dataset']
    for key in ('SMPLNodes','CamPose'):
        if key in cfg.model:del cfg.model[key]
    cfg.data.data_root=str(processed);cfg.data.scene_idx=int(sequence)
    cfg.data.start_timestep=0;cfg.data.end_timestep=79
    cfg.trainer.optim.num_iters=iters
    path=scratch/'fit_config.yaml';OmegaConf.save(cfg,path)
    return path


def timed_phase(label,action,record):
    start=time.monotonic();status='failed'
    try:
        result=action();status='completed';return result
    finally:
        from unrealityview.data.cache import atomic_json
        data=json.loads(record.read_text()) if record.exists() else {}
        data[label]=dict(seconds=time.monotonic()-start,status=status);atomic_json(record,data)


def fit_scene(upstream,native,processed,sequence,scratch,iters,env,timing_record):
    timed_phase('preprocessing',lambda:subprocess.run([sys.executable,'datasets/preprocess.py','--dataset','pandaset',
        '--data_root',str(native),'--target_dir',str(processed),'--workers','1',
        '--scene_ids',sequence,'--process_keys','images','lidar','calib','pose','dynamic_masks','objects'],
        cwd=upstream,env=env,check=True),timing_record)
    timed_phase('sky_masks',lambda:subprocess.run([sys.executable,str(ROOT/'tools/chase_3dgs/extract_sky_masks.py'),
        '--scene',str(processed/sequence)],env=env,check=True),timing_record)
    cfg=build_config(upstream,processed,sequence,scratch,iters)
    timed_phase('fit_15000' if iters==15000 else 'fit',lambda:subprocess.run([sys.executable,'tools/train.py','--config_file',str(cfg),
        '--output_root',str(scratch/'fits'),'--project','chase','--run_name',sequence],
        cwd=upstream,env=env,check=True),timing_record)
    return scratch/'fits/chase'/sequence/'checkpoint_final.pth'


def render_frame_inputs(c2w,k,time,index,device):
    import torch
    from datasets.base.pixel_source import get_rays
    h,w=192,320
    pose=torch.as_tensor(c2w,dtype=torch.float32,device=device)
    intrinsic=torch.as_tensor(k,dtype=torch.float32,device=device)
    x,y=torch.meshgrid(torch.arange(w,device=device),torch.arange(h,device=device),indexing='xy')
    origins,directions,norm=get_rays(x.flatten(),y.flatten(),pose,intrinsic)
    image=dict(origins=origins.reshape(h,w,3),viewdirs=directions.reshape(h,w,3),direction_norm=norm.reshape(h,w,1),
        pixel_coords=torch.stack((y.float()/h,x.float()/w),-1),
        normed_time=torch.full((h,w),float(time),device=device),
        img_idx=torch.full((h,w),index*6,dtype=torch.long,device=device),
        frame_idx=torch.full((h,w),index,dtype=torch.long,device=device))
    camera=dict(camera_to_world=pose,intrinsics=intrinsic,height=torch.tensor(h,device=device),width=torch.tensor(w,device=device))
    return image,camera


def save_gif(images,path,timestamps):
    duration=1000*float(np.median(np.diff(timestamps)))
    images[0].save(path,save_all=True,append_images=images[1:],duration=duration,loop=0)


def cleanup_difix(raw,reference,pins):
    source=ROOT/'.cache/third_party/difix/src'
    sys.path.insert(0,str(source))
    from pipeline_difix import DifixPipeline
    spec=pins['difix_model']
    pipe=DifixPipeline.from_pretrained(spec['repo'],revision=spec['revision'],code_revision=spec['revision'],use_safetensors=True,trust_remote_code=True).to('cuda')
    output=[]
    for image,ref in zip(raw,reference):
        fixed=pipe('remove degradation',image=image,ref_image=ref,num_inference_steps=1,
                   timesteps=[199],guidance_scale=0.,width=image.width,height=image.height).images[0]
        if fixed.size!=image.size:raise ValueError('Difix changed output geometry')
        output.append(fixed.convert('RGB'))
    return output


def render_scene(upstream,processed,native,sequence,checkpoint,output,difix=False):
    import torch
    from omegaconf import OmegaConf
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str
    pins=json.loads((ROOT/'tools/chase_3dgs/upstream_pins.json').read_text())
    cfg=OmegaConf.load(checkpoint.parent/'config.yaml');device=torch.device('cuda')
    dataset=DrivingDataset(data_cfg=cfg.data)
    trainer=import_str(cfg.trainer.type)(**cfg.trainer,num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,num_train_images=len(dataset.train_image_set),num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,scene_aabb=dataset.get_aabb().reshape(2,3),device=device)
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint),load_only_model=True);trainer.set_eval()
    directory=output/'pandaset'/sequence;directory.mkdir(parents=True,exist_ok=True)
    (directory/'rgb').mkdir(exist_ok=True);(directory/'alpha').mkdir(exist_ok=True)
    timestamps=json.loads((native/sequence/'camera/front_camera/timestamps.json').read_text())
    if len(timestamps)!=80 or dataset.num_img_timesteps!=80:raise ValueError('Expected all 80 exact native frames')
    native_camera=dataset.pixel_source.camera_data[0]
    references=[];renders=[];coverages=[];ground_coverages=[];ego_coverages=[];heldout=[];calibrations=[]
    with torch.no_grad():
        if dataset.test_image_set is not None:
            for i in range(len(dataset.test_image_set)):
                image,camera=dataset.test_image_set[i]
                image={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in image.items()}
                camera={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in camera.items()}
                prediction=trainer(image,camera)['rgb'].float().clamp(0,1)
                mse=(prediction-image['pixels']).square().mean().item()
                heldout.append(float(-10*np.log10(max(mse,1e-12))))
        # Held-out PSNR above uses the original test split. Only the novel
        # view pass below uses consistent mean appearance for all 80 frames.
        with consistent_novel_appearance(trainer,80):
            for i in range(80):
                front=processed_front_pose(processed/sequence,i)
                np.testing.assert_allclose(front,native_camera.cam_to_worlds[i].cpu().numpy(),atol=1e-4)
                pose,k=chase_calibration(front)
                image,camera=render_frame_inputs(pose,k,native_camera.normalized_time[i],i,device)
                result=trainer(image,camera,novel_view=True)
                rgb=result['rgb'].detach().cpu().float().clamp(0,1).numpy()
                alpha=result['opacity'].detach().cpu().float().squeeze(-1).clamp(0,1).numpy()
                rendered=Image.fromarray(np.rint(rgb*255).astype(np.uint8))
                rendered.save(directory/'rgb'/f'{i:05d}.png');renders.append(rendered)
                Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(directory/'alpha'/f'{i:05d}.png')
                coverage=alpha_coverages(alpha)
                coverages.append(coverage['whole']);ground_coverages.append(coverage['ground']);ego_coverages.append(coverage['ego'])
                with Image.open(native/sequence/'camera/front_camera'/f'{i:02d}.jpg') as ref:references.append(ref.convert('RGB').resize((320,192)))
                calibrations.append(dict(camera_to_world=pose.tolist(),intrinsics=k.tolist()))
    cleaned=cleanup_difix(renders,references,pins) if difix else None
    if cleaned:
        (directory/'rgb_clean').mkdir(exist_ok=True)
        for i,frame in enumerate(cleaned):frame.save(directory/'rgb_clean'/f'{i:05d}.png')
        save_gif(cleaned,directory/'chase_difix.gif',timestamps)
    save_gif(renders,directory/'chase_raw.gif',timestamps)
    # Projection audit at 15/25/40m in front-local FLU. These are calibration
    # reference markers, not predicted planner output or measured lane positions.
    from unrealityview.data.real_world import CV_TO_FLU
    columns=(0,39,79);rows=3 if cleaned else 2
    sheet=Image.new('RGB',(960,rows*220),'#111820');draw=ImageDraw.Draw(sheet)
    for col,i in enumerate(columns):
        views=[references[i],renders[i].copy()]+([cleaned[i].copy()] if cleaned else [])
        front=processed_front_pose(processed/sequence,i);flu=front.copy();flu[:3,:3]=front[:3,:3]@CV_TO_FLU.T
        points=(flu@np.array([[15,0,0,1],[25,0,0,1],[40,0,0,1]]).T).T[:,:3]
        pose,k=chase_calibration(front);uv=project_world(points,pose,k)
        for image in views[1:]:
            painter=ImageDraw.Draw(image)
            for x,y in uv:painter.ellipse((x-3,y-3,x+3,y+3),fill='#258cff')
        labels=['Original front dashcam','Raw dynamic 3DGS + calibration markers']+(['Difix cleanup + calibration markers'] if cleaned else [])
        for row,(label,image) in enumerate(zip(labels,views)):
            draw.text((col*320+5,row*220+5),f'{label} / {i}',fill='white');sheet.paste(image,(col*320,row*220+28))
    sheet.save(directory/'inspection.png')
    quality=dict(source_archive_sha256=json.loads((native/'.archive.json').read_text())['archive_sha256'],
        fit_iterations=int(cfg.trainer.optim.num_iters),heldout_psnr_db=float(np.mean(heldout)) if heldout else 0.,heldout_samples=len(heldout),
        heldout_definition='Every tenth native timestep withheld from all six image training views',
        chase_alpha_coverage_min=min(coverages),chase_alpha_coverage_mean=float(np.mean(coverages)),
        chase_ground_coverage_min=min(ground_coverages),chase_ground_coverage_mean=float(np.mean(ground_coverages)),
        per_frame_ground_alpha=ground_coverages,novel_appearance_policy='mean_embedding_all_frames',gate_policy=GATE_POLICY,
        alpha_definition='Raw Gaussian opacity>=0.5; whole frame and fixed lower band rows67:192 measured separately; sky is not geometry',
        ego_footprint_coverage_min=min(ego_coverages),per_frame_alpha=coverages,
        difix_enabled=difix,upstream=pins,calibration_reference='Blue markers at local FLU (15,0,0),(25,0,0),(40,0,0); not planner output')
    manifest=export_manifest(directory,sequence,timestamps,quality,frame_directory='rgb_clean' if cleaned else 'rgb',
        teacher='DriveStudio OmniRe dynamic 3DGS'+(' + Difix generative cleanup' if cleaned else ''))
    (directory/'calibration.json').write_text(json.dumps(calibrations,indent=2)+'\n')
    if manifest is None:raise RuntimeError('Dynamic teacher numeric gate failed; diagnostics saved, importer manifest withheld')
    return directory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',required=True);p.add_argument('--processed',required=True)
    p.add_argument('--sequence',required=True);p.add_argument('--output',default='outputs/chase_teacher')
    p.add_argument('--upstream',default='.cache/third_party/drivestudio');p.add_argument('--scratch',default='.cache/chase_3dgs')
    p.add_argument('--iters',type=int,default=15000);p.add_argument('--checkpoint');p.add_argument('--difix-cleanup',action='store_true');a=p.parse_args()
    if len(a.sequence)!=3 or not a.sequence.isdigit():p.error('sequence must be three digits')
    if not 100<=a.iters<=30000:p.error('iters must be 100..30000')
    from tools.chase_3dgs.runtime_env import configure_runtime_tools
    configure_runtime_tools()
    upstream=Path(a.upstream).resolve();pins=json.loads((ROOT/'tools/chase_3dgs/upstream_pins.json').read_text())
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=pins['drivestudio']['commit']:raise ValueError('Unpinned DriveStudio checkout')
    native=Path(a.native).resolve();processed=Path(a.processed).resolve();scratch=Path(a.scratch).resolve()/a.sequence
    stale=Path(a.output).resolve()/'pandaset'/a.sequence/'manifest.json'
    if stale.exists():stale.unlink()
    scratch.mkdir(parents=True,exist_ok=True);processed.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PYTHONPATH':f'{upstream}:{ROOT}/src:{ROOT}'}
    sys.path.insert(0,str(upstream))
    timing_record=Path(a.output).resolve()/'pandaset'/a.sequence/'timings.json'
    checkpoint=Path(a.checkpoint).resolve() if a.checkpoint else fit_scene(upstream,native,processed,a.sequence,scratch,a.iters,env,timing_record)
    timed_phase('render_and_metrics',lambda:render_scene(upstream,processed,native,a.sequence,checkpoint,Path(a.output).resolve(),a.difix_cleanup),timing_record)


if __name__=='__main__':main()
