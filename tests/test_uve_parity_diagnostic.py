import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from scripts.diagnose_uve_parity import diagnose,encode_mode,read_cameras,per_camera,tensor_sha256
from unrealityview.models.mini_uve import MiniUVE
from unrealityview.data.cache import save_episode


class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.ones(1))
    def forward_features(self,x):
        return {'x_norm_patchtokens':(x.mean((1,2,3))[:,None,None]*self.weight).expand(-1,160,384)}


def test_read_only_modes_ray_decomposition_and_hashes(tmp_path):
    n=17;encoder=MiniUVE(backbone=Backbone());path=tmp_path/'episode.safetensors'
    episode=dict(uve_tokens=torch.zeros(n,192,384,dtype=torch.float16),
        chase_latents=torch.zeros(n,4,24,40,dtype=torch.float16),ego_states=torch.zeros(n,3),
        camera_mask=torch.ones(n,6,dtype=torch.bool),intrinsics=torch.eye(3).repeat(n,6,1,1),camera_to_ego=torch.eye(4).repeat(n,6,1,1))
    from scripts.reencode_uve_cache import SLOTS
    for j,name in enumerate(SLOTS):
        folder=tmp_path/'raw'/name;folder.mkdir(parents=True)
        for frame in (0,8,16):Image.fromarray(np.full((140,224,3),40+j*20,dtype=np.uint8)).save(folder/f'{frame:05d}.jpg')
    # Only sampled frames need raw files; the source tensors stay unchanged.
    cameras=read_cameras(tmp_path,[0,8,16]);values=encode_mode(encoder,cameras,episode,[0,8,16],torch.device('cpu'),'fp32',1)
    episode['uve_tokens'][[0,8,16]]=values['full'].half()
    save_episode(tmp_path,episode,dict(source_id='pandaset/040',split='train'))
    before=path.read_bytes()
    result=diagnose(path,encoder,torch.device('cpu'),frame_count=3,precisions=('fp32',))
    assert result['source_unchanged'] and path.read_bytes()==before
    assert result['sampled_frame_ids']==[0,8,16]
    assert result['fingerprints']['ray_projection']['sha256']==tensor_sha256(encoder.ray_projection)
    assert all(x['frozen_parity_limits_pass'] for x in result['modes'].values())
    assert all(x['relative_rms']==0 for rows in result['pairwise'].values() for x in rows)
    assert result['modes']['cpu_fp32_batch1']['ray_subtracted_cached_vs_semantic'][0]['relative_rms']<.01
    json.dumps(result,allow_nan=False)
    controlled=diagnose(path,encoder,torch.device("cpu"),frame_count=3,precisions=("fp16","bf16"),fp32_rays=True)
    assert all(r["status"]=="completed" for r in controlled["modes"].values())
    assert controlled["ray_policy"]=="controlled_FP32_ray_construction"


def test_masked_cameras_and_nonfinite_rejection():
    a=torch.ones(2,192,384);b=a.clone();b.reshape(2,6,32,384)[:,1]=99
    mask=torch.ones(2,6,dtype=torch.bool);mask[:,1]=False
    rows=per_camera(a,b,mask)
    assert rows[1]['available_frames']==0
    assert all(r.get('relative_rms',0)==0 for r in rows)
    with pytest.raises(ValueError,match='Nonfinite'):per_camera(a,b*float('nan'),mask)


def test_disabled_autocast_is_not_reported_as_requested_precision(monkeypatch):
    from contextlib import nullcontext
    from scripts.diagnose_uve_parity import precision_context
    monkeypatch.setattr(torch,'autocast',lambda *args,**kwargs:nullcontext())
    with pytest.raises(NotImplementedError,match='disabled'):
        with precision_context(torch.device('cpu'),'bf16'):pass
