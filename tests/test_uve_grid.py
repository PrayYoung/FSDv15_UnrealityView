import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from unrealityview.models.mini_uve import MiniUVE
from unrealityview.geometry import plucker_rays
from unrealityview.data.cache import save_episode,load_episode,validate_episode,episode_grid,require_uve_grid
from unrealityview.data.dataset import CachedEpisodeChunkDataset
from scripts.reencode_uve_cache import encode_episode,reencode,SLOTS,parity


class Backbone(nn.Module):
    def forward_features(self,x):
        return {'x_norm_patchtokens':x.mean((1,2,3))[:,None,None].expand(-1,160,384)}


@pytest.mark.parametrize('grid',[(4,8),(5,8),(10,16)])
def test_shapes_geometry_and_camera_mask(grid):
    model=MiniUVE(backbone=Backbone(),token_grid=grid)
    cameras=torch.ones(1,6,3,140,224)
    k=torch.eye(3).repeat(1,6,1,1);ext=torch.eye(4).repeat(1,6,1,1)
    plain=model(cameras);geometry=model(cameras,k,ext)
    expected=(plucker_rays(k,ext,height=grid[0],width=grid[1])@model.ray_projection).reshape(1,-1,384)
    assert plain.shape==(1,6*grid[0]*grid[1],384)
    assert torch.allclose(geometry-plain,expected,atol=1e-6)
    mask=torch.tensor([[1,0,1,1,1,1]],dtype=torch.bool)
    masked=model(cameras,k,ext,mask).reshape(1,6,-1,384)
    assert masked[:,1].eq(0).all()
    assert torch.equal(masked[:,0],geometry.reshape(1,6,-1,384)[:,0])


def tensors(grid,n=48):
    return dict(uve_tokens=torch.ones(n,6*grid[0]*grid[1],384,dtype=torch.float16),
        chase_latents=torch.zeros(n,4,24,40,dtype=torch.float16),ego_states=torch.zeros(n,3),
        timestamps=torch.arange(n,dtype=torch.float64)/10,camera_timestamps=torch.arange(n,dtype=torch.float64)[:,None].expand(-1,6)/10,
        camera_mask=torch.ones(n,6,dtype=torch.bool),intrinsics=torch.eye(3).repeat(n,6,1,1),camera_to_ego=torch.eye(4).repeat(n,6,1,1))


def test_dynamic_cache_and_mixed_grid_rejection(tmp_path):
    for i,g in enumerate([(4,8),(10,16)]):
        data=tensors(g);assert validate_episode(data)==48
        save_episode(tmp_path/'train'/str(i),data,{'source_id':f'pandaset/{i}','uve_grid':g})
    with pytest.raises(ValueError,match='Mixed'):CachedEpisodeChunkDataset(tmp_path,'train')
    with pytest.raises(ValueError,match='Missing'):episode_grid(tensors((5,8)),{})
    assert episode_grid(tensors((4,8)),{})==(4,8)
    with pytest.raises(ValueError,match='mismatch'):require_uve_grid({},SimpleNamespace(uve_grid=(10,16)))
    require_uve_grid({'uve_grid':[10,16]},SimpleNamespace(uve_grid=(10,16)))


def test_reencoder_parity_slot_order_and_preserves_source(tmp_path):
    source=tmp_path/'source';scene=source/'train/first';scene.mkdir(parents=True)
    data=tensors((4,8),n=17);save_episode(scene,data,{'source_id':'pandaset/040','split':'train'})
    for slot,name in enumerate(SLOTS):
        (scene/'raw'/name).mkdir(parents=True)
        for i in range(17):Image.fromarray(np.full((140,224,3),20+slot*30,dtype=np.uint8)).save(scene/'raw'/name/f'{i:05d}.jpg')
    (scene/'raw/chase').mkdir();(scene/'raw/chase/00000.jpg').write_bytes(b'preserved target')
    original=encode_episode(scene/'episode.safetensors',MiniUVE(backbone=Backbone()),torch.device('cpu'))
    save_episode(scene,original,{'source_id':'pandaset/040','split':'train'})
    before=(scene/'episode.safetensors').read_bytes()
    for name in ('manifest.json','split_manifest.json'):(source/name).write_text('{}')
    output=tmp_path/'encoded'
    reencode(source,output,(10,16),torch.device('cpu'),tmp_path/'parity.json',
        encoder=MiniUVE(backbone=Backbone(),token_grid=(10,16)),parity_encoder=MiniUVE(backbone=Backbone()))
    assert json.loads((tmp_path/'parity.json').read_text())['passed']
    result=load_episode(output/'train/first/episode.safetensors')
    assert result['uve_tokens'].shape==(17,960,384)
    for key,value in original.items():
        if key!='uve_tokens':assert torch.equal(value,result[key])
    assert before==(scene/'episode.safetensors').read_bytes()
    assert (output/'train/first/raw/chase/00000.jpg').read_bytes()==b'preserved target'
    wrong={**original,'uve_tokens':original['uve_tokens'].reshape(17,6,32,384).flip(1).reshape(17,192,384)}
    assert not parity(original,wrong)['passed']


def test_checkpoint_grid_roundtrip_and_legacy_default(tmp_path):
    from unrealityview.checkpoint import save_checkpoint,load_checkpoint
    from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
    from unrealityview.models.driving_policy_head import DrivingPolicyHead
    model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4))
    model.uve_grid=(10,16)
    policy=DrivingPolicyHead(hidden_dim=16,depth=1,num_heads=4)
    save_checkpoint(tmp_path,model,policy,500)
    loaded,_,config=load_checkpoint(tmp_path)
    assert loaded.uve_grid==(10,16) and config['uve_grid']==[10,16]
    with pytest.raises(ValueError,match='mismatch'):require_uve_grid(config,SimpleNamespace(uve_grid=(4,8)))
    config.pop('uve_grid');(tmp_path/'config.json').write_text(json.dumps(config))
    loaded,_,_=load_checkpoint(tmp_path);assert loaded.uve_grid==(4,8)


def test_native_and_exact_two_by_two_pooling_preserve_patch_order():
    class Spatial(nn.Module):
        def forward_features(self,x):return {'x_norm_patchtokens':torch.arange(160).float()[None,:,None].expand(len(x),-1,384)}
    images=torch.zeros(1,6,3,140,224)
    native=MiniUVE(backbone=Spatial(),token_grid=(10,16))(images).reshape(1,6,10,16,384)
    pooled=MiniUVE(backbone=Spatial(),token_grid=(5,8))(images).reshape(1,6,5,8,384)
    assert torch.equal(native[0,0,:,:,0],torch.arange(160).reshape(10,16))
    expected=native.reshape(1,6,5,2,8,2,384).mean((3,5))
    assert torch.equal(pooled,expected)
