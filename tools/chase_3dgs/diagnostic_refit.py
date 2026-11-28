"""One 20k-update scene040 refit; two poses from the same immutable checkpoint."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from unrealityview.data.cache import atomic_json
from tools.chase_3dgs.diagnostic_common import FRAMES,SIZE,chase_pose,seal_views


def run(native,processed,scratch,work,output,upstream,sequence='040',pose_names=('A','B'),psnr_threshold=26.):
    import torch
    from omegaconf import OmegaConf
    from tools.chase_3dgs.runtime_env import configure_runtime_tools
    configure_runtime_tools()
    from tools.chase_3dgs.render_chase_teacher import fit_scene,render_frame_inputs
    from tools.chase_3dgs.contract import processed_front_pose
    from tools.chase_3dgs.render_policy import consistent_novel_appearance,alpha_coverages
    root=Path.cwd();scratch.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    pins=json.loads((root/'tools/chase_3dgs/upstream_pins.json').read_text())
    actual=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if actual!=pins['drivestudio']['commit']:raise ValueError('Unpinned DriveStudio')
    sys.path.insert(0,str(upstream));env={**os.environ,'PYTHONPATH':f'{upstream}:{root}/src:{root}'}
    checkpoint=fit_scene(upstream,native,processed,sequence,scratch,20000,env,output/'fit_timings.json',root/'configs/scene040_diagnostic.yaml')
    cfg=OmegaConf.load(checkpoint.parent/'config.yaml')
    assert int(cfg.model.Background.init.near_randoms)==10000
    assert int(cfg.model.Background.init.from_lidar.num_samples)==800000
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str
    dataset=DrivingDataset(data_cfg=cfg.data);device=torch.device('cuda')
    trainer=import_str(cfg.trainer.type)(**cfg.trainer,num_timesteps=dataset.num_img_timesteps,model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),num_full_images=len(dataset.full_image_set),test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2,3),device=device)
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint),load_only_model=True);trainer.set_eval()
    heldout=[];groups={f'pose_{name}':{} for name in pose_names};references={};calibrations=[]
    from PIL import Image
    from tools.chase_3dgs.native_scene import NativeScene
    native_scene=NativeScene(native,sequence=sequence);start=time.monotonic();coverage={name:[] for name in pose_names}
    from unrealityview.data.real_world import pose_matrix
    positions=np.stack([pose_matrix(p)[:3,3] for p in native_scene.metadata[0][2]])
    distance=np.r_[0,np.linalg.norm(np.diff(positions,axis=0),axis=1).cumsum()]
    atomic_json(output/'trajectory_check.json',dict(front_reference_distance_from_start_m=distance.tolist(),first_rendered_frame=12,distance_at_first_rendered_frame_m=float(distance[12]),note='Recorded trajectory distance is descriptive;it does not prove chase visibility or surface observation.'))
    with torch.no_grad():
        if dataset.test_image_set is None:raise ValueError('Held-out native views missing')
        for index in range(len(dataset.test_image_set)):
            image,camera=dataset.test_image_set[index]
            image={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in image.items()}
            camera={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in camera.items()}
            predicted=trainer(image,camera)['rgb'].float().clamp(0,1)
            heldout.append(-10*np.log10(max(float((predicted-image['pixels']).square().mean()),1e-12)))
        atomic_json(output/'native_quality.json',dict(heldout_psnr_db=float(np.mean(heldout)),heldout_samples=len(heldout),threshold_db=psnr_threshold,passed=float(np.mean(heldout))>=psnr_threshold,not_novel_view_ground_truth=True))
        with consistent_novel_appearance(trainer,80):
            for index in FRAMES:
                front=processed_front_pose(processed/sequence,index)
                np.testing.assert_allclose(front,dataset.pixel_source.camera_data[0].cam_to_worlds[index].cpu().numpy(),atol=1e-4)
                ref=work/'reference'/f'{index:05d}.png';ref.parent.mkdir(exist_ok=True)
                native_scene.front(index,SIZE).save(ref);references[index]=str(ref.relative_to(work))
                for name in pose_names:
                    pose,k=chase_pose(front,name);image,camera=render_frame_inputs(pose,k,dataset.pixel_source.camera_data[0].normalized_time[index],index,device,size=SIZE)
                    result=trainer(image,camera,novel_view=True);rgb=result['rgb'].float().clamp(0,1).cpu().numpy()
                    alpha=result['opacity'].float().squeeze(-1).clamp(0,1).cpu().numpy()
                    # Coverage band scaled to match original rows67:192 contract.
                    small=np.asarray(Image.fromarray(np.rint(alpha*255).astype('uint8')).resize((320,192)))/255
                    coverage[name].append(alpha_coverages(small))
                    path=work/f'pose_{name}'/f'{index:05d}.png';path.parent.mkdir(exist_ok=True)
                    Image.fromarray(np.rint(rgb*255).astype('uint8')).save(path);groups[f'pose_{name}'][index]=str(path.relative_to(work))
                    calibrations.append(dict(index=index,pose=name,camera_to_world=pose.tolist(),intrinsics=k.tolist(),size=list(SIZE)))
                atomic_json(output/'progress.json',dict(stage='rendering',completed_frames=index-11,total_frames=68))
    atomic_json(output/'render_metrics.json',dict(render_seconds=time.monotonic()-start,coverage=coverage,near_randoms=10000,far_randoms=int(cfg.model.Background.init.far_randoms),lidar_init=800000,configured_iterations=20000,checkpoint_shared_between_poses=True))
    atomic_json(output/'calibration.json',calibrations)
    seal_views(work,groups,references,native_scene.timestamps,dict(experiment='refit',sequence=sequence,source_archive=json.loads((native/'.archive.json').read_text()),heldout_gate_is_not_visual_approval=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--processed',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream',type=Path,default=Path('.cache/third_party/drivestudio'))
    a=p.parse_args();run(**{k:v.resolve() for k,v in vars(a).items()})
