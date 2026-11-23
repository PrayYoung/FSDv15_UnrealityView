import copy
import json
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from unrealityview.models.lora import (LoRALinear,LoRAWeightManager,inject_dit_lora,
    extract_lora_state_dict,save_lora,load_lora,model_fingerprint)
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.driving_policy_head import DrivingPolicyHead
from unrealityview.objectives import meanflow_loss
from unrealityview.data.stylize_cache import TemporalStylizer,spatial_shader,StylizedChunkDataset
from unrealityview.data.cache import save_episode,atomic_json
from safetensors.torch import save_file


def model():
    torch.manual_seed(12)
    net=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=32,depth=8,num_heads=4,latent_shape=(2,4,6)))
    nn.init.normal_(net.output.weight,std=.05)
    for block in net.blocks:nn.init.normal_(block.mod[-1].weight,std=.02)
    return net


def inputs():
    return dict(z=torch.randn(2,4,2,4,6),r=torch.tensor([.2,.1]),t=torch.tensor([.8,.6]),
                z_prev=torch.randn(2,4,2,4,6),uve_tokens=torch.randn(2,4,192,384),stage_ages=torch.zeros(2,4))


def test_lora_zero_initialization_freezing_and_jvp():
    base=model();example=inputs();expected=base(**example).detach()
    inject_dit_lora(base,rank=4,start_block=4)
    torch.testing.assert_close(base(**example),expected,atol=0,rtol=0)
    assert all(not p.requires_grad for block in base.blocks[:4] for p in block.parameters())
    assert all("lora_" in name for name,p in base.named_parameters() if p.requires_grad)
    assert not any(isinstance(m,LoRALinear) for m in base.cross[:2].modules())
    loss,_=meanflow_loss(base,example["z"],example["z_prev"],example["uve_tokens"],fm_ratio=0,jvp_method="exact")
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in base.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for name,p in base.named_parameters() if "lora_b" in name)
    assert all(p.grad is None for name,p in base.named_parameters() if "lora_" not in name)
    state=extract_lora_state_dict(base)
    assert all(name.endswith((".lora_a",".lora_b",".scale")) for name in state)


def test_merged_output_hybrid_and_exact_restoration(tmp_path):
    base=model();clean=copy.deepcopy(base);example=inputs();original=clean(**example).detach()
    inject_dit_lora(base,rank=4,start_block=4)
    for module in base.modules():
        if isinstance(module,LoRALinear):nn.init.normal_(module.lora_b,std=.02)
    state=extract_lora_state_dict(base)
    expected=base(**example).detach()
    manager=LoRAWeightManager(clean)
    result=manager.apply_adapters([("comic",state,1.)])
    torch.testing.assert_close(clean(**example),expected,atol=1e-5,rtol=1e-4)
    assert result["merge_ms"]>=0 and not result["cache_hit"]
    assert manager.apply_adapters([("comic",state,1.)])["cache_hit"]
    assert not any(isinstance(m,LoRALinear) for m in clean.modules())
    manager.apply_adapters([("comic",state,.55),("cyberpunk",state,.55)])
    hybrid={name:layer.weight.detach().clone() for name,layer in clean.named_modules() if isinstance(layer,nn.Linear)}
    manager.apply_adapters([("comic",state,1.1)])
    for name,layer in clean.named_modules():
        if name in hybrid:torch.testing.assert_close(layer.weight,hybrid[name],atol=1e-7,rtol=1e-6)
    for _ in range(10):manager.apply_adapters([("comic",state,1.5)]);manager.restore_base()
    torch.testing.assert_close(clean(**example),original,atol=0,rtol=0)
    save_lora(tmp_path/"comic.safetensors",base,"comic",3)
    loaded=load_lora(tmp_path/"comic.safetensors",manager.base_sha256)
    assert loaded.size_bytes<100_000 and loaded.style=="comic"
    manager.apply_adapters([("comic",loaded,1.)])
    torch.testing.assert_close(clean(**example),expected,atol=1e-5,rtol=1e-4)
    with pytest.raises(ValueError,match="different base"):load_lora(tmp_path/"comic.safetensors","wrong")


def test_invalid_merge_is_atomic_and_planner_is_isolated():
    base=model();adapter_model=copy.deepcopy(base);inject_dit_lora(adapter_model,rank=4,start_block=0)
    for module in adapter_model.modules():
        if isinstance(module,LoRALinear):nn.init.normal_(module.lora_b,std=.02)
    state=extract_lora_state_dict(adapter_model);manager=LoRAWeightManager(base)
    policy=DrivingPolicyHead(hidden_dim=32,num_heads=4,depth=1).eval()
    tokens=torch.randn(2,192,384);before=policy(tokens).detach();fingerprint=model_fingerprint(policy)
    for strength in (0.,.5,1.,1.5):manager.apply_adapters([("style",state,strength)])
    torch.testing.assert_close(policy(tokens),before,atol=0,rtol=0)
    assert model_fingerprint(policy)==fingerprint
    current={k:v.clone() for k,v in base.state_dict().items()}
    invalid={k:v.clone() for k,v in state.items()};next(iter(invalid.values())).fill_(float("nan"))
    with pytest.raises(ValueError):manager.apply_adapters([("invalid",invalid,1.)])
    for key,value in base.state_dict().items():torch.testing.assert_close(value,current[key],atol=0,rtol=0)
    for strength in (-.01,1.51,float("nan")):
        with pytest.raises(ValueError):manager.apply_adapters([("invalid",state,strength)])


@pytest.mark.parametrize("style",["comic","cyberpunk"])
def test_temporal_shader_is_deterministic_and_bounded(style):
    pytest.importorskip("cv2")
    image=np.zeros((32,48,3),np.uint8);image[:16]=[80,140,210];image[16:]=[70,70,70];image[12:25,15:28]=[190,40,50]
    one,two=TemporalStylizer(style),TemporalStylizer(style)
    for offset in (0,1,2,3):
        frame=np.roll(image,offset,axis=1)
        a,b=one(frame),two(frame)
        np.testing.assert_array_equal(a,b)
        assert a.dtype==np.uint8 and a.shape==image.shape and np.isfinite(a).all()
    one.reset();np.testing.assert_array_equal(one(image),two.__class__(style)(image))
    with pytest.raises(ValueError):spatial_shader(image,"invalid")


def test_style_dataset_uses_separate_latents_and_real_previous_mixture(tmp_path):
    import hashlib
    base_root=tmp_path/"base";style_root=tmp_path/"style"
    directory=base_root/"train"/"sequence"
    real={"uve_tokens":torch.randn(64,192,384),"chase_latents":torch.zeros(64,4,24,40),"ego_states":torch.zeros(64,3)}
    save_episode(directory,real,{"source":"pandaset","source_id":"pandaset/unit-fixture","fps":10,"frames":64})
    path=directory/"episode.safetensors";before=hashlib.sha256(path.read_bytes()).hexdigest()
    styled=style_root/"comic/train/sequence";styled.mkdir(parents=True)
    save_file({"style_latents":torch.ones_like(real["chase_latents"])},str(styled/"style.safetensors"))
    atomic_json(styled/"metadata.json",dict(base_sha256=before,style="comic",split="train"))
    real_previous=StylizedChunkDataset(base_root,style_root,"comic",real_previous_ratio=1.)[0]
    style_previous=StylizedChunkDataset(base_root,style_root,"comic",real_previous_ratio=0.)[0]
    assert real_previous["z_prev"].sum()==0 and real_previous["z0"].min()==1
    assert style_previous["z_prev"].min()==1
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert real_previous["uve_tokens"].shape==(4,192,384)
