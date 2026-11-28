"""Pinned cleanup plus causal three-timestamp features and compact scene cache."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from unrealityview.data.cache import atomic_json
from unrealityview.data.flying_cache import FRAME_IDS,CAMERAS,SIZES,pack_mask,encode_images,write_scene
from unrealityview.explicit3d.project import project,align_camera_scales,composite_live_first,pinhole_close_once
from unrealityview.explicit3d.preprocess import preprocess_rgb
from tools.chase_3dgs.diagnostic_common import SIZE,chase_pose,comparison,load_pil
from tools.chase_3dgs.native_scene import NativeScene
from tools.chase_3dgs.flying_plan import INSPECTION_FRAMES,resize_geometry,resize_pair


def rgb_image(value):
    return Image.fromarray(np.rint(value.detach().cpu().float().clamp(0,1).numpy()*255).astype('uint8'))


def mask_image(value):return Image.fromarray(value.cpu().numpy().astype('uint8')*255)


def encode(vae,images):
    array=np.stack([np.asarray(i.convert('RGB')) for i in images])
    x=torch.from_numpy(array.copy()).permute(0,3,1,2).cuda().float()/255
    result=vae.encode(x*2-1).latents
    if not torch.isfinite(result).all():raise ValueError('Nonfinite TAESD features')
    return result.half().cpu()


def band_holes(mask):return dict(whole=1-float(mask.float().mean()),lower_band=1-float(mask[134:].float().mean()))


def run(native,processed,work,output,sequence,split,recipe):
    from diffusers import AutoencoderTiny
    from tools.chase_3dgs.explicit_proof import load_depth
    from tools.chase_3dgs.render_chase_teacher import cleanup_difix
    from tools.chase_3dgs.diagnostic_common import FRAMES
    torch.manual_seed(42);torch.set_num_threads(8)
    views=json.loads((work/'views.json').read_text());pins=json.loads(Path('tools/chase_3dgs/upstream_pins.json').read_text())
    if views['frames']!=FRAME_IDS or views['extra']['sequence']!=sequence:raise ValueError('Teacher frame identity mismatch')
    quality=json.loads((output/'quality.json').read_text())
    if not quality['passed']:raise ValueError('Numeric teacher gate must pass before feature export')
    raw=[load_pil(work/views['groups']['pose_B'][str(i)]) for i in FRAMES]
    reference=[load_pil(work/views['references'][str(i)]) for i in FRAMES]
    timings={};cuda_ms=[];start=time.monotonic()
    def progress(stage,details):
        if stage=='model_loaded':timings['model_load_seconds']=details['seconds']
        if 'cuda_elapsed_ms' in details:cuda_ms.append(details['cuda_elapsed_ms'])
        if stage!='frame' or details['completed_frames']%10==0:
            atomic_json(output.parents[1]/'progress.json',dict(stage='difix_'+stage,scene=sequence,**details))
    cleaned=cleanup_difix(raw,reference,pins,progress=progress,cuda_timing=True)
    timings['cleanup_seconds']=time.monotonic()-start;timings['cuda_ms']=cuda_ms
    atomic_json(output/'cleanup_timings.json',timings)
    del raw;gc.collect();torch.cuda.empty_cache()
    model,depth_pins=load_depth();ae_pin=json.loads(Path('configs/pixart_stateless_pins.json').read_text())['taesd']
    vae=AutoencoderTiny.from_pretrained(ae_pin['repo'],revision=ae_pin['revision'],use_safetensors=True).cuda().eval().requires_grad_(False)
    scene=NativeScene(native,use_lidar=False,sequence=sequence)
    origin=np.loadtxt(processed/sequence/'ego_pose/000.txt')
    teacher_calibration={row['index']:row for row in json.loads((output/'calibration.json').read_text())}
    values={key:[] for key in ('camera_latents','camera_to_world','camera_intrinsics','camera_timestamps','target_to_world','timestamps')}
    images={};masks={};resize_metadata={};rows=[];cache={};columns=[{} for _ in range(4)]
    for key in SIZES:
        for role in ('target','rough'):
            values[f'{role}_latents_{key}']=[];images[f'{role}_rgb_{key}']=[]
        values[f'target_intrinsics_{key}']=[]
        for name in ('M','observed_M'):masks[f'{name}_{key}_packed']=[]
    def tensor(value):return torch.as_tensor(value,device='cuda',dtype=torch.float32)
    feature_start=time.monotonic();depth_calls=0
    with torch.inference_mode():
        for index in range(2,80):
            frame_start=time.monotonic();multi,poses_np,offsets=scene.cameras_multires(index,(SIZE,(256,160)))
            rgb,k_np=multi[SIZE];camera_rgb,camera_k=multi[(256,160)]
            colors=tensor(rgb)/255;k=tensor(k_np);poses=tensor(poses_np)
            batch=preprocess_rgb(colors,392)
            with torch.autocast('cuda',dtype=torch.float16):prediction=model(batch)
            depth=torch.nn.functional.interpolate(prediction.float()[:,None],(384,640),mode='bilinear',align_corners=True)[:,0]
            depth_calls+=1
            if not torch.isfinite(depth).all():raise ValueError('Nonfinite FP16 depth')
            aligned,scales,edges=align_camera_scales(depth,k,poses);cache[index]=(colors,aligned,k,poses)
            if index<12:continue
            target_np,target_k=chase_pose(poses_np[0],'B')
            teacher_pose=origin@np.asarray(teacher_calibration[index]['camera_to_world'])
            np.testing.assert_allclose(target_np,teacher_pose,atol=1e-4)
            np.testing.assert_allclose(target_k,teacher_calibration[index]['intrinsics'],atol=1e-4)
            lag5=float(scene.timestamps[index]-scene.timestamps[index-5]);lag10=float(scene.timestamps[index]-scene.timestamps[index-10])
            if not (.44<=lag5<=.56 and .94<=lag10<=1.06):raise ValueError('Unexpected causal camera time gap')
            live,m0,_=project(*cache[index],tensor(target_np),tensor(target_k),SIZE)
            previous5,m5,_=project(*cache[index-5],tensor(target_np),tensor(target_k),SIZE)
            previous10,m10,_=project(*cache[index-10],tensor(target_np),tensor(target_k),SIZE)
            two,m2=composite_live_first(live,m0,previous5,m5)
            three,observed=composite_live_first(two,m2,previous10,m10)
            closed,mask,filled=pinhole_close_once(three,observed)
            deviation=float((closed[m0]-live[m0]).abs().max()) if m0.any() else 0.
            if deviation!=0:raise ValueError('Live pixel identity lost')
            rough_image=rgb_image(closed);observed_image=mask_image(observed);closed_image=mask_image(mask)
            values['camera_latents'].append(encode(vae,[Image.fromarray(x) for x in camera_rgb]))
            values['camera_to_world'].append(torch.from_numpy(poses_np).float());values['camera_intrinsics'].append(torch.from_numpy(camera_k).float())
            values['camera_timestamps'].append(torch.tensor([float(scene.timestamps[index])+d for d in offsets],dtype=torch.float64))
            values['target_to_world'].append(torch.from_numpy(target_np).float());values['timestamps'].append(torch.tensor(float(scene.timestamps[index]),dtype=torch.float64))
            clean=cleaned[index-12]
            for key,size in SIZES.items():
                resized_k,geometry=resize_geometry(SIZE,size,target_k);resize_metadata[key]=geometry
                teacher_image=resize_pair(clean,geometry);rough=resize_pair(rough_image,geometry)
                for role,image in (('target',teacher_image),('rough',rough)):
                    path=work/'paired_png'/key/role/f'{index:05d}.png';path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
                    images[f'{role}_rgb_{key}'].append(image)
                encoded=encode(vae,[teacher_image,rough]);values[f'target_latents_{key}'].append(encoded[0]);values[f'rough_latents_{key}'].append(encoded[1])
                values[f'target_intrinsics_{key}'].append(torch.from_numpy(resized_k).float())
                for name,image in (('M',closed_image),('observed_M',observed_image)):
                    binary=np.asarray(resize_pair(image,geometry,mask=True))>127
                    masks[f'{name}_{key}_packed'].append(pack_mask(binary))
            if index in INSPECTION_FRAMES:
                for column,image in zip(columns,(Image.fromarray(rgb[0]),rough_image,closed_image.convert('RGB'),clean)):column[index]=image
            rows.append(dict(index=index,lag5_seconds=lag5,lag10_seconds=lag10,single=band_holes(m0),two_timestamp=band_holes(m2),three_timestamp=band_holes(observed),pinhole_closed=band_holes(mask),pinhole_filled_fraction=float(filled.float().mean()),live_max_deviation=deviation,camera_scales=scales.cpu().tolist(),overlap_edges=edges,frame_wall_including_io_encoding_export_ms=(time.monotonic()-frame_start)*1000))
            for old in list(cache):
                if old<index-10:del cache[old]
            atomic_json(output.parents[1]/'progress.json',dict(stage='paired_features',scene=sequence,completed_frames=len(rows),total_frames=68))
    tensors={name:torch.stack(parts) for name,parts in values.items()}
    tensors['frame_indices']=torch.tensor(FRAME_IDS,dtype=torch.int64)
    for name,parts in masks.items():tensors[name]=torch.stack(parts)
    pixel_hashes={}
    for name,items in images.items():
        blob,offsets,hashes=encode_images(items);tensors[name+'_webp']=blob;tensors[name+'_offsets']=offsets;pixel_hashes[name]=hashes
    comparison(columns,['Native front','3-ts+pinhole rough','M: projected+filled','PoseB Difix teacher'],frames=INSPECTION_FRAMES).save(output/'inspection.png')
    import diffusers
    metadata=dict(schema='unrealityview.flying.v1',recipe_hash=recipe,sequence=sequence,source_id='pandaset/'+sequence,split=split,frame_indices=FRAME_IDS,camera_names=CAMERAS,source_camera_reference='Native full-resolution JPEG decoded once per sourceframe/camera;640x384and256x160resized directly from original',camera_rgb_size=[256,160],camera_latent_shape=[6,4,20,32],target_sizes=SIZES,resize_geometry=resize_metadata,mask_encoding='np.packbits little-endian on width;binary after unpack',M_definition='pinhole_closed_M=projection_union OR onepass>=4of8neighbor fill;observed_M separately preserves only actual source projections',rgb_encoding='lossless WebP inside compact safetensors;decoded pixel hashes verified;canonical paired PNGs saved in remote scratch',rgb_pixel_sha256=pixel_hashes,depth_model=depth_pins,taesd=ae_pin,teacher_model=pins['difix_model'],teacher_source=pins['drivestudio'],runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,diffusers=diffusers.__version__,numpy=np.__version__),teacher_recipe=dict(near_randoms=10000,configured_iterations=20000,pose='B',native_render=[640,384]),student_lidar_used=False,student_z_prev_used=False,depth_forward_calls=depth_calls,source_frame_indices=list(range(2,80)),cache_policy='Steady state one6-camera depth pass;real t>t-5>t-10 priority;ten-frame preroll',temporal_note='Stale content can remain in currently unobserved regions;current-covered identity is not universal ghosting rejection',quality=quality,visual_approval=False)
    metadata['source_archive']=json.loads(Path('configs/pandaset_archive.json').read_text())
    metadata['teacher_offline_full_scene_used']=True
    metadata['student_frontend_is_causal']=True
    atomic_json(output/'feature_metrics.json',dict(frames=rows,feature_seconds=time.monotonic()-feature_start,depth_forward_calls=depth_calls,configured_input_size=392,all_live_identity_checks_passed=all(r['live_max_deviation']==0 for r in rows)))
    write_scene(output,tensors,metadata,quality)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('native','processed','work','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('sequence','split','recipe'):p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
