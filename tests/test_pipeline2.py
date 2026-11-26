import inspect,math
import torch
import pytest
from torch import nn
from unrealityview.pipeline2.model import StatelessPixArt,fourier_rays
from unrealityview.pipeline2.losses import pseudo_huber,charbonnier,three_term_loss
from unrealityview.pipeline2.data import causal_pair,RecoveryScenes
from unrealityview.geometry import plucker_rays


def test_causal_lag_and_reserved_split(tmp_path):
    assert causal_pair(1)==(0,1) and causal_pair(79)==(78,79)
    with pytest.raises(ValueError):causal_pair(0)
    with pytest.raises(ValueError,match='Reserved'):RecoveryScenes(tmp_path,'test')
    assert 'z_prev' not in inspect.signature(StatelessPixArt.forward).parameters


def test_fourier_full_pose_and_clean_loss_math():
    k=torch.tensor([[100.,0,112],[0,100,70],[0,0,1.]])
    pose=torch.eye(4);pose[:3,3]=torch.tensor([1.,2.,3.])
    rays=plucker_rays(k,pose,10,16)
    assert rays.shape==(160,6) and fourier_rays(rays).shape==(160,54)
    moved=pose.clone();moved[2,3]+=2
    assert not torch.allclose(rays,plucker_rays(k,moved,10,16))
    x=torch.tensor([0.,1.],requires_grad=True);y=torch.zeros_like(x)
    expected=.1**2*(math.sqrt(101)-1)/2
    assert pseudo_huber(x,y).item()==pytest.approx(expected)
    assert charbonnier(y,y).item()==pytest.approx(.001)
    perceptual=lambda a,b:(a-b).abs().mean()
    loss,terms=three_term_loss(x,y,x,y,perceptual)
    assert torch.allclose(loss,.5*terms['pseudo_huber']+terms['charbonnier']+.8*terms['vgg'])
    loss.backward();assert torch.isfinite(x.grad).all()


def test_real_diffusers_tiny_stateless_tokens_gradients_and_fusion():
    from diffusers import PixArtTransformer2DModel
    torch.manual_seed(9)
    base=PixArtTransformer2DModel(num_attention_heads=2,attention_head_dim=8,num_layers=1,
        cross_attention_dim=16,caption_channels=16,sample_size=32,in_channels=4,out_channels=8,
        patch_size=2,norm_type='ada_norm_single',use_additional_conditions=False)
    model=StatelessPixArt(base,rank=2,alpha=4)
    args=(torch.randn(1,4,24,40),torch.randn(1,6,4,20,32),torch.randn(1,6,160,6),torch.randn(1,240,6))
    model.train();out=model(*args);assert out.shape==(1,4,24,40)
    out.square().mean().backward()
    assert model.role_emb.weight.grad is not None and model.ray_mlp[-1].weight.grad is not None
    assert base.pos_embed.proj.weight.grad is None
    assert base.transformer_blocks[0].attn1.to_q.lora_b.grad is not None
    # Make a nonzero trained adapter delta before verifying fused inference.
    with torch.no_grad():base.transformer_blocks[0].attn1.to_q.lora_b.normal_(std=.01)
    model.eval()
    with torch.no_grad():before=model(*args);repeat=model(*args);after=model.fuse_lora()(*args)
    assert torch.equal(before,repeat)
    torch.testing.assert_close(before,after,rtol=2e-5,atol=2e-5)


def test_vgg_loss_preserves_pixel_gradients_with_frozen_features():
    from unrealityview.pipeline2.losses import ChannelNormalizedVGG16
    features=nn.Sequential(*[nn.Conv2d(3,3,1) if i in (0,5,10,18) else nn.ReLU() for i in range(23)])
    metric=ChannelNormalizedVGG16(features)
    pred=torch.rand(2,3,8,8,requires_grad=True);target=torch.rand_like(pred)
    metric(pred,target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert all(p.grad is None for p in features.parameters())


def test_actual_frozen_taesd_decoder_allows_prediction_gradients():
    from diffusers import AutoencoderTiny
    ae=AutoencoderTiny().eval().requires_grad_(False)
    z=torch.randn(1,4,2,2,requires_grad=True)
    ae.decode(z).sample.square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert all(p.grad is None for p in ae.parameters())
