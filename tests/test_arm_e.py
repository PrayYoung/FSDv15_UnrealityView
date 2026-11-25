import copy
import json
import subprocess
from pathlib import Path
import pytest
import torch
from torch import nn
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.driving_policy_head import DrivingPolicyHead
from unrealityview.models.blocks import Attention
from unrealityview.data.dataset import stage_uve_tokens
from unrealityview.checkpoint import save_checkpoint,load_checkpoint
from unrealityview.objectives import meanflow_loss
from unrealityview.models.lora import inject_dit_lora,LoRAWeightManager


def tiny(geometry=True):
    model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4,latent_shape=(2,4,6),geometry_conditioning=geometry,qk_logit_scale=geometry))
    torch.nn.init.normal_(model.output.weight,std=.02)
    for b in model.blocks:torch.nn.init.normal_(b.mod[-1].weight,std=.02)
    return model


def inputs():
    return torch.randn(2,4,2,4,6),torch.randn(2,4,8,390),torch.zeros(2,4)


def test_default_forward_is_bitwise_legacy_and_loads_strictly(tmp_path):
    import importlib.util,sys
    spec=importlib.util.spec_from_file_location('arm_e_legacy',Path(__file__).parent/'fixtures/arm_e_legacy.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    for architecture in ('block','stage'):
        config=dict(hidden_dim=16,depth=4,num_heads=4,latent_shape=(2,4,6),cross_attention=architecture)
        torch.manual_seed(7);old=module.ConditionalMeanFlowDiT(module.DiTConfig(**config))
        torch.nn.init.normal_(old.output.weight,std=.1)
        torch.manual_seed(7);new=ConditionalMeanFlowDiT(DiTConfig(**config));new.load_state_dict(old.state_dict(),strict=True)
        z=torch.randn(1,4,2,4,6);tokens=torch.randn(1,4,8,384);r=torch.tensor([.1]);t=torch.tensor([.7])
        assert torch.equal(old(z,r,t,z,tokens),new(z,r,t,z,tokens))
        save_checkpoint(tmp_path/architecture,new,DrivingPolicyHead(hidden_dim=16,depth=1,num_heads=4),3)
        p=tmp_path/architecture/'config.json';data=json.loads(p.read_text());data['dit'].pop('geometry_conditioning');data['dit'].pop('qk_logit_scale');p.write_text(json.dumps(data))
        loaded,_,_=load_checkpoint(p.parent);assert torch.equal(loaded(z,r,t,z,tokens),new(z,r,t,z,tokens))


def test_masked_ray_corruption_and_jvp_dropout_consistency(monkeypatch):
    model=tiny();z,tokens,ages=inputs();tokens[:,:,0,:384]=0
    corrupted=tokens.clone();corrupted[:,:,0,384:]=float('nan')
    r=torch.tensor([.1,.2]);t=torch.tensor([.6,.7])
    assert torch.equal(model(z,r,t,z,tokens,ages),model(z,r,t,z,corrupted,ages))
    from unrealityview import objectives
    original=objectives.directional_derivative;seen=[];forward=[]
    def spy(*args,**kwargs):seen.append(args[5].detach().clone());return original(*args,**kwargs)
    # args: model,z,r,t,velocity,previous,tokens,ages,backend
    monkeypatch.setattr(objectives,'directional_derivative',spy)
    handle=model.register_forward_pre_hook(lambda _,args:forward.append(args[3].detach().clone()))
    loss,_=meanflow_loss(model,z,z,tokens,ages,jvp_method='exact',fm_ratio=0.,prev_chunk_dropout=.25)
    handle.remove();assert seen and torch.equal(seen[0],forward[-1])
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    clone=copy.deepcopy(model);assert torch.equal(clone(z,r,t,z,tokens,ages),model(z,r,t,z,tokens,ages))
    for block in model.blocks:assert not hasattr(block.attn,'logit_scale') and hasattr(block.cross_attn,'logit_scale')


def test_stage_packing_and_lora_checkpoint(tmp_path):
    episode=dict(uve_tokens=torch.ones(4,192,384),intrinsics=torch.eye(3).repeat(4,6,1,1),camera_to_ego=torch.eye(4).repeat(4,6,1,1),camera_mask=torch.ones(4,6,dtype=torch.bool))
    episode['camera_mask'][:,1]=False;episode['uve_tokens'][:,32:64]=0
    packed=stage_uve_tokens(episode,torch.tensor([0,2]),include_rays=True)
    assert packed.shape==(2,192,390) and packed[:,32:64].eq(0).all()
    assert torch.equal(stage_uve_tokens(episode,[0,2]),episode['uve_tokens'][[0,2]])
    model=tiny();save_checkpoint(tmp_path,model,DrivingPolicyHead(hidden_dim=16,depth=1,num_heads=4),3)
    loaded,_,_=load_checkpoint(tmp_path);assert loaded.config.geometry_conditioning
    manager=LoRAWeightManager(copy.deepcopy(loaded));inject_dit_lora(loaded,rank=2,start_block=0)
    z,tokens,ages=inputs();output=loaded(z,torch.zeros(2),torch.ones(2),z,tokens,ages)
    assert torch.isfinite(output).all() and manager is not None
