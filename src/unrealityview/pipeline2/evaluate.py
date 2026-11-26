"""All eight unseen validation scenes, conditioning probes and fixed-noise clips."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from unrealityview.data.cache import atomic_json
from unrealityview.pipeline2.losses import sobel_l1
from unrealityview.pipeline2.data import rgb


def image(tensor):return Image.fromarray((tensor.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8'))


def panel(rows,labels):
    out=Image.new('RGB',(320*len(rows),216))
    for i,(row,label) in enumerate(zip(rows,labels)):
        out.paste(image(row),(320*i,24));ImageDraw.Draw(out).text((320*i+5,5),label,fill='white')
    return out


@torch.no_grad()
def metrics(pred,gt,vgg,device,batch=8):
    if not torch.isfinite(pred).all():raise ValueError('Nonfinite evaluation pixels')
    mse=(pred-gt).square().flatten(1).mean(1)
    psnr=(-10*torch.log10(mse.clamp_min(1e-12))).mean().item();perceptual=edge=0.
    for start in range(0,len(pred),batch):
        x=pred[start:start+batch].to(device);y=gt[start:start+batch].to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):perceptual+=float(vgg(x,y))*len(x)
        edge+=float(sobel_l1(x,y))*len(x)
    return dict(psnr_db=psnr,vgg_distance=perceptual/len(pred),sobel_edge_l1=edge/len(pred))


@torch.no_grad()
def arm_e_frames(directory,scene,vae,device):
    from unrealityview.checkpoint import load_checkpoint
    from unrealityview.data.dataset import stage_context,stage_uve_tokens
    model,policy,config=load_checkpoint(directory,device);del policy
    model.eval();frames=[]
    for start in range(8,80,8):
        previous=scene['chase_latents'][start-8:start].permute(1,0,2,3)[None].to(device).float()
        ids,ages=stage_context(scene,start,10)
        tokens=stage_uve_tokens(scene,ids,(4,8),include_rays=config['dit'].get('geometry_conditioning',False))[None].to(device)
        noise=torch.randn(previous.shape,device=device,generator=torch.Generator(device=device).manual_seed(917+start))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model.sample_chunk(previous,tokens,1,ages[None].to(device),noise=noise)
            decoded=(vae.decode(pred[0].permute(1,0,2,3)).sample+1)/2
        frames.append(decoded.float().clamp(0,1).cpu())
    del model;torch.cuda.empty_cache()
    return torch.cat(frames)


@torch.no_grad()
def evaluate_pilot(model,vae,vgg,val,device,output,arm_e,v1_adapter=None,pins=None):
    scene_reports=[];example=None
    if v1_adapter is None or pins is None:raise ValueError('The v2 comparison requires preserved v1 weights')
    from diffusers import PixArtTransformer2DModel
    from safetensors.torch import load_file
    from unrealityview.pipeline2.model import StatelessPixArt
    spec=pins['transformer']
    base=PixArtTransformer2DModel.from_pretrained(spec['repo'],revision=spec['revision'],subfolder='transformer',use_safetensors=True,torch_dtype=torch.bfloat16)
    v1=StatelessPixArt(base).to(device,dtype=torch.bfloat16)
    for parameter in v1.parameters():
        if parameter.requires_grad:parameter.data=parameter.data.float()
    v1.restore_adapter(load_file(str(v1_adapter))).eval().fuse_lora()
    for scene,identity in enumerate(val.ids):
        truth=torch.stack([rgb(val.directories[scene].parent/'raw/chase'/f'{frame:05d}.jpg') for frame in range(1,80)])
        predictions={name:[] for name in ('all_6_cams','drop_front','swapped_scene_cams','fresh_noise','codec','v1_baseline')}
        fixed=torch.randn(1,4,24,40,device=device,generator=torch.Generator(device=device).manual_seed(917+scene))
        fresh=torch.Generator(device=device).manual_seed(1271+scene)
        for start in range(1,80,8):
            pairs=[(scene,frame) for frame in range(start,min(start+8,80))];noise=fixed.expand(len(pairs),-1,-1,-1)
            for variant in ('all_6_cams','drop_front','swapped_scene_cams','fresh_noise','v1_baseline'):
                cams,camr,tr,z,gt=val.batch(pairs,device,variant=variant if variant in ('drop_front','swapped_scene_cams') else 'all_6_cams',target_ray_frame='target_ego_v1' if variant=='v1_baseline' else 'previous_ego')
                eps=torch.randn(z.shape,device=device,generator=fresh) if variant=='fresh_noise' else noise
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=(v1 if variant=='v1_baseline' else model)(eps,cams,camr,tr);pixels=(vae.decode(pred).sample+1)/2
                    if variant=='all_6_cams':codec=(vae.decode(z).sample+1)/2
                predictions[variant].append(pixels.float().clamp(0,1).cpu())
                if variant=='all_6_cams':predictions['codec'].append(codec.float().clamp(0,1).cpu())
                if example is None:example=(eps[:1].clone(),cams[:1],camr[:1],tr[:1])
        predictions={k:torch.cat(v) for k,v in predictions.items()}
        baseline=arm_e_frames(arm_e,val.episodes[scene],vae,device)
        row={'source_id':identity,'frames':79,'variants':{k:metrics(v,truth,vgg,device) for k,v in predictions.items()}}
        row['matched_frames_8_79']={'arm_e':metrics(baseline,truth[7:],vgg,device),
            'pipeline2':metrics(predictions['all_6_cams'][7:],truth[7:],vgg,device),
            'v1_baseline':metrics(predictions['v1_baseline'][7:],truth[7:],vgg,device),
            'codec':metrics(predictions['codec'][7:],truth[7:],vgg,device),
            'note':'ArmEusesgroundtruthprevious8framelatents;Pipeline2hasnohistoryinput'}
        row['temporal']={}
        for name in ('all_6_cams','fresh_noise','v1_baseline'):
            delta=predictions[name][1:]-predictions[name][:-1];true_delta=truth[1:]-truth[:-1]
            row['temporal'][name]={'frame_delta_l1':float(delta.abs().mean()),'motion_residual_l1':float((delta-true_delta).abs().mean()),'consecutive_frames':79}
        scene_reports.append(row);atomic_json(output/'per_scene.json',scene_reports)
        # Frame16 is shared by both model protocols. No synthetic visuals.
        directory=output/'inspection';directory.mkdir(exist_ok=True)
        labels=['GT native front','TAESD codec','Arm E: GT history','PixArt v1: preserved','PixArt v2: corrected']
        panel([truth[15],predictions['codec'][15],baseline[8],predictions['v1_baseline'][15],predictions['all_6_cams'][15]],labels).save(directory/(identity.replace('/','_')+'.png'))
        if scene==0:
            frames=[panel([truth[i-1],predictions['codec'][i-1],baseline[i-8],predictions['v1_baseline'][i-1],predictions['all_6_cams'][i-1]],labels) for i in range(8,40)]
            frames[0].save(output/'rollout_fixed_noise.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    variants={key:{metric:float(np.mean([r['variants'][key][metric] for r in scene_reports])) for metric in ('psnr_db','vgg_distance','sobel_edge_l1')} for key in predictions}
    matched={key:{metric:float(np.mean([r['matched_frames_8_79'][key][metric] for r in scene_reports])) for metric in ('psnr_db','vgg_distance','sobel_edge_l1')} for key in ('arm_e','pipeline2','v1_baseline','codec')}
    temporal={key:{metric:float(np.mean([r['temporal'][key][metric] for r in scene_reports])) for metric in ('frame_delta_l1','motion_residual_l1')} for key in ('all_6_cams','fresh_noise','v1_baseline')}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(10):model(*example)
        torch.cuda.synchronize();clock=time.perf_counter()
        for _ in range(30):model(*example)
        torch.cuda.synchronize();ms=(time.perf_counter()-clock)*1000/30
        torch.cuda.synchronize();clock=time.perf_counter()
        for _ in range(30):vae.decode(model(*example))
        torch.cuda.synchronize();decoded_ms=(time.perf_counter()-clock)*1000/30
    report=dict(status='completed',recipe_version='prompt_d_corrected_v2_ego_motion',vgg_metric='weighted_channel_sum_normalized_squared_L2_v2',val_scenes=8,frames_per_scene=79,variants=variants,matched_frames_8_79=matched,
        temporal=temporal,latency=dict(fused_lora_model_ms=ms,fused_lora_model_fps=1000/ms,model_plus_decode_ms=decoded_ms,includes_camera_encoding=False),
        historical_references=dict(codec_claim_db=30.25,v2_db=21.8846,arm_e_db=21.5858,arm_e_f0_db=24.,note='Historicalreferencesuseddifferentwindowselection;matchedArmEandcodecabovearemeasuredonthesameframes'),
        differences={key:variants[key]['psnr_db']-variants['all_6_cams']['psnr_db'] for key in ('drop_front','swapped_scene_cams')},
        v1_parity=dict(reference_psnr_db=25.116267919540405,reproduced_psnr_db=variants['v1_baseline']['psnr_db'],tolerance_db=.05,passed=abs(variants['v1_baseline']['psnr_db']-25.116267919540405)<=.05),
        conditioning_note='V2 target rays use observed target-frame planar ego motion;V1 retains original target-local rays. RGB lag and GT frames are matched;conditioning differs.',
        quality_acceptance='Pendingmanualvisualreview;donotstartstylesorflyingviewtraining')
    atomic_json(output/'evaluation.json',report)
