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
    expected=(math.sqrt(1.01)-.1)/2
    assert pseudo_huber(x,y).item()==pytest.approx(expected)
    assert charbonnier(y,y).item()==pytest.approx(.001)
    legacy=(.1**2*(torch.sqrt(1+((x-y)/.1).square())-1)).mean()
    torch.testing.assert_close(pseudo_huber(x,y),legacy/.1)
    perceptual=lambda a,b:(a-b).abs().mean()
    loss,terms=three_term_loss(x,y,x,y,perceptual)
    assert torch.allclose(terms['charbonnier'],2*charbonnier(x,y))
    assert torch.allclose(loss,.5*terms['pseudo_huber']+terms['charbonnier']+.8*terms['vgg'])
    loss.backward();assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('additional_conditions',[False,True])
def test_real_diffusers_tiny_stateless_tokens_gradients_and_fusion(additional_conditions):
    from diffusers import PixArtTransformer2DModel
    torch.manual_seed(9)
    heads=3 if additional_conditions else 2
    width=heads*8
    base=PixArtTransformer2DModel(num_attention_heads=heads,attention_head_dim=8,num_layers=1,
        cross_attention_dim=width,caption_channels=width,sample_size=128 if additional_conditions else 32,in_channels=4,out_channels=8,
        patch_size=2,norm_type='ada_norm_single',use_additional_conditions=additional_conditions)
    model=StatelessPixArt(base,rank=2,alpha=4)
    assert torch.count_nonzero(model.role_emb.weight)==0
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


def test_vgg_channel_sum_squared_distance_layer_weights_and_clamp():
    from unrealityview.pipeline2.losses import ChannelNormalizedVGG16
    class Zero(nn.Module):
        def forward(self,x):return x*0
    metric=ChannelNormalizedVGG16(nn.Sequential(*([nn.Identity() for _ in range(22)]+[Zero()])))
    mean=metric.mean;std=metric.std
    x=mean+std*torch.tensor([.1,0.,0.])[None,:,None,None]
    y=mean+std*torch.tensor([0.,.1,0.])[None,:,None,None]
    # First three taps have orthogonal unit features (squared distance2),
    # final tap is zero; layer weighting gives(2+2+2+.5*0)/3.5.
    assert metric(x,y).item()==pytest.approx(6/3.5,rel=1e-5)
    out_of_range=torch.tensor([-.5,1.5,.5])[None,:,None,None].requires_grad_()
    torch.testing.assert_close(metric(out_of_range,y),metric(out_of_range.clamp(0,1),y))
    metric(out_of_range,y).backward()
    assert out_of_range.grad[0,0].eq(0).all() and out_of_range.grad[0,1].eq(0).all()


def test_training_can_sample_clean_and_all_augmented_camera_banks(tmp_path,monkeypatch):
    from PIL import Image
    ds=RecoveryScenes.__new__(RecoveryScenes)
    ds.episodes=[{'intrinsics':torch.eye(3).expand(80,6,3,3).clone(),
                  'camera_to_ego':torch.eye(4).expand(80,6,4,4).clone(),
                  'chase_latents':torch.zeros(80,4,24,40)}]
    ds.directories=[tmp_path/'episode.safetensors']
    (tmp_path/'raw/chase').mkdir(parents=True)
    Image.new('RGB',(320,192)).save(tmp_path/'raw/chase/00002.jpg')
    ds.latents=[torch.stack([torch.full((80,6,4,20,32),float(i)) for i in range(4)])]
    def choose(low,high,size):
        assert low==0 and high==4 and size==(6,)
        return torch.tensor([0,1,2,3,0,1])
    monkeypatch.setattr(torch,'randint',choose)
    monkeypatch.setattr(torch,'rand',lambda *a,**k:torch.tensor(1.))
    cameras,*_=ds.batch([(0,2)],'cpu',train=True)
    torch.testing.assert_close(cameras[0,:,0,0,0],torch.tensor([0.,1.,2.,3.,0.,1.]))
