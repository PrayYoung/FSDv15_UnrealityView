"""Train/validation-only cached raw cameras with strict one-frame causal lag."""
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from unrealityview.data.cache import load_episode
from unrealityview.data.cache_recovery import RAW_CAMERAS
from unrealityview.data.splits import stable_split
from unrealityview.geometry import plucker_rays
from safetensors import safe_open

CAMERAS=RAW_CAMERAS[:6]
TRAIN_TENSORS=('chase_latents','intrinsics','camera_to_ego','camera_mask',
               'timestamps','camera_timestamps','ego_states')


def load_training_episode(path):
    """Unused train UVE features are not needed in a transfer-only cache."""
    with safe_open(str(path),framework='pt',device='cpu') as source:
        episode={name:source.get_tensor(name) for name in TRAIN_TENSORS}
    shapes={'chase_latents':(80,4,24,40),'intrinsics':(80,6,3,3),
            'camera_to_ego':(80,6,4,4),'camera_mask':(80,6),
            'timestamps':(80,),'camera_timestamps':(80,6),'ego_states':(80,3)}
    for name,shape in shapes.items():
        if tuple(episode[name].shape)!=shape or not torch.isfinite(episode[name]).all():
            raise ValueError('Invalid camera transfer tensor: '+name)
    return episode


def previous_from_target(previous,target):
    """Planar cached world(x,y,yaw) motion, expressed in previous ego FLU."""
    delta=target[..., :2]-previous[..., :2];yaw=previous[...,2]
    dx=yaw.cos()*delta[...,0]+yaw.sin()*delta[...,1]
    dy=-yaw.sin()*delta[...,0]+yaw.cos()*delta[...,1]
    turn=target[...,2]-yaw
    transform=torch.zeros((*previous.shape[:-1],4,4),dtype=previous.dtype,device=previous.device)
    transform[...,0,0]=turn.cos();transform[...,0,1]=-turn.sin()
    transform[...,1,0]=turn.sin();transform[...,1,1]=turn.cos()
    transform[...,0,3]=dx;transform[...,1,3]=dy
    transform[...,2,2]=1;transform[...,3,3]=1
    return transform


def causal_pair(frame,frames=80):
    if not 1<=frame<frames:raise ValueError('Target requires an earlier camera frame')
    return frame-1,frame


def rgb(path):
    with Image.open(path) as im:array=np.array(im.convert('RGB'),copy=True)
    return torch.from_numpy(array).permute(2,0,1).float()/255


class RecoveryScenes:
    def __init__(self,root,split,expected_count=None):
        if split not in ('train','val'):raise ValueError('Reserved test scenes cannot be opened')
        self.split=split;self.directories=sorted((Path(root)/split).glob('*/episode.safetensors'))
        if expected_count is not None and len(self.directories)!=expected_count:raise ValueError('Scene count mismatch')
        self.episodes=[];self.ids=[];self.latents=[]
        for path in self.directories:
            meta=json.loads(path.with_name('metadata.json').read_text());identity=meta['source_id']
            if meta['split']!=split or stable_split('pandaset',identity.split('/')[-1],42)!=split:raise ValueError('Scene split mismatch')
            if meta.get('target_view')!='native_front' or meta['frames']!=80:raise ValueError('Expected80frame native-front recovery cache')
            transfer=meta.get('pipeline2_transfer')
            if transfer and hashlib.sha256(path.read_bytes()).hexdigest()!=transfer['sha256']:
                raise ValueError('Transfer tensor fingerprint mismatch')
            episode=load_training_episode(path) if split=='train' else load_episode(path)
            if episode['camera_to_ego'].shape!=(80,6,4,4):raise ValueError('Missing full camera calibration')
            if not episode['camera_mask'].all():raise ValueError('Incomplete camera snapshot')
            if (episode['camera_timestamps'][:-1].max(1).values>episode['timestamps'][1:]).any():
                raise ValueError('Lagged camera snapshot contains future capture')
            for camera in (*CAMERAS,'chase'):
                if any(not (path.parent/'raw'/camera/f'{i:05d}.jpg').is_file() for i in range(80)):raise ValueError('Missing raw cache RGB')
            self.episodes.append(episode);self.ids.append(identity)
        if len(set(self.ids))!=len(self.ids):raise ValueError('Duplicate scene identity')

    @torch.no_grad()
    def encode(self,vae,device,batch_size=48,augmentation_banks=3):
        # Independent RGB jitter banks are encoded once, then sampled per camera
        # during training; validation always uses the clean bank.
        banks=1+augmentation_banks if self.split=='train' else 1
        for path in self.directories:
            images=torch.stack([rgb(path.parent/'raw'/camera/f'{frame:05d}.jpg') for frame in range(80) for camera in CAMERAS])
            encoded=[]
            for bank in range(banks):
                rows=[]
                for start in range(0,len(images),batch_size):
                    x=images[start:start+batch_size].to(device)
                    x=F.interpolate(x,size=(160,256),mode='bilinear',align_corners=False)
                    if bank:
                        contrast=1+torch.empty(len(x),1,1,1,device=device).uniform_(-.1,.1)
                        brightness=1+torch.empty_like(contrast).uniform_(-.1,.1)
                        mean=x.mean((-2,-1),keepdim=True);x=((x-mean)*contrast+mean)*brightness
                        x=x.clamp(0,1)
                    with torch.autocast('cuda',dtype=torch.bfloat16):z=vae.encode(x*2-1).latents
                    rows.append(z.half().cpu())
                encoded.append(torch.cat(rows).reshape(80,6,4,20,32))
            self.latents.append(torch.stack(encoded))

    def batch(self,pairs,device,train=False,variant='all_6_cams',target_ray_frame='previous_ego'):
        cams=[];camera_rays=[];target_rays=[];targets=[];truth=[]
        for scene,frame in pairs:
            previous,target=causal_pair(frame);episode=self.episodes[scene]
            donor=(scene+1)%len(self.episodes) if variant=='swapped_scene_cams' else scene
            if donor==scene and variant=='swapped_scene_cams':raise ValueError('Different-scene donor required')
            source=self.episodes[donor];banks=self.latents[donor]
            if train:
                choice=torch.randint(0,len(banks),(6,))
                latent=torch.stack([banks[choice[c],previous,c] for c in range(6)])
            else:latent=banks[0,previous].clone()
            if variant=='drop_front' or (train and torch.rand(())<.1):
                index=1 if variant=='drop_front' else int(torch.randint(6,()).item());latent[index]=0
            cams.append(latent)
            # Rays retain full calibrated camera orientation and translation;
            # there is no projection onto a flat road or learned depth input.
            camera_rays.append(plucker_rays(source['intrinsics'][previous].float(),source['camera_to_ego'][previous].float(),10,16,image_size=(140,224)))
            target_pose=episode['camera_to_ego'][target,1].float()
            if target_ray_frame=='previous_ego':
                target_pose=previous_from_target(episode['ego_states'][previous].float(),episode['ego_states'][target].float())@target_pose
            elif target_ray_frame!='target_ego_v1':raise ValueError('Unknown target ray convention')
            target_rays.append(plucker_rays(episode['intrinsics'][target,1].float(),target_pose,12,20,image_size=(140,224)))
            targets.append(episode['chase_latents'][target].float())
            truth.append(rgb(self.directories[scene].parent/'raw/chase'/f'{target:05d}.jpg'))
        return tuple(torch.stack(values).to(device) for values in (cams,camera_rays,target_rays,targets,truth))
