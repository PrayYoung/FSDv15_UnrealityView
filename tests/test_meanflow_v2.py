import copy
import hashlib
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from unrealityview.models.blocks import TimeEmbed
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.driving_policy_head import DrivingPolicyHead
from unrealityview.objectives import directional_derivative,meanflow_loss,adaptive_loss,sample_times
from unrealityview.training import clip_training_gradients,initialize_weights
from unrealityview.checkpoint import save_checkpoint,load_checkpoint
from unrealityview.styles import validate_style_slug


def tiny(scale=10.):
    model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4,latent_shape=(2,4,6),time_scale=scale))
    torch.nn.init.normal_(model.output.weight,std=.02)
    for block in model.blocks:torch.nn.init.normal_(block.mod[-1].weight,std=.02)
    return model


def test_time_scale_derivative_and_fp32_jvp():
    torch.manual_seed(8)
    low=TimeEmbed(16,10.);high=TimeEmbed(16,1000.);high.load_state_dict(low.state_dict())
    # Same sinusoidal phase -> identical network activations; only chain rule differs.
    _,a=torch.func.jvp(low,(torch.tensor([.5]),),(torch.ones(1),))
    _,b=torch.func.jvp(high,(torch.tensor([.005]),),(torch.ones(1),))
    torch.testing.assert_close(b,a*100,rtol=3e-5,atol=1e-4)
    # Whole-network derivatives also depend on phase: no fixed 100x assertion.
    model=tiny();z=torch.randn(1,4,2,4,6);v=torch.randn_like(z);tokens=torch.randn(1,4,8,384)
    r=torch.tensor([.1]);t=torch.tensor([.6]);ages=torch.zeros(1,4)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        exact=directional_derivative(model,z,r,t,v,z,tokens,ages,'exact')
    finite=directional_derivative(model,z,r,t,v,z,tokens,ages,'finite',epsilon=1e-3)
    assert exact.dtype==torch.float32 and not exact.requires_grad
    torch.testing.assert_close(exact,finite,rtol=.03,atol=.003)


@pytest.mark.parametrize('nfe',[1,2,4])
def test_constant_velocity_oracle(nfe):
    class Oracle(ConditionalMeanFlowDiT):
        def forward(self,z,r,t,*args):return torch.ones_like(z)*.25
    model=Oracle(DiTConfig(hidden_dim=16,depth=4,num_heads=4,latent_shape=(2,4,6)))
    previous=torch.zeros(1,4,2,4,6);noise=torch.randn_like(previous)
    prediction=model.sample_chunk(previous,torch.zeros(1,4,2,384),nfe,noise=noise)
    torch.testing.assert_close(prediction,noise-.25)


def test_bounded_loss_and_sampling_and_diagnostics():
    mse=torch.tensor([1.,1e30,1e100],dtype=torch.float64,requires_grad=True)
    loss=adaptive_loss(mse);assert 0<=loss<=1;loss.backward();assert torch.isfinite(mse.grad).all()
    r,t=sample_times(1000,'cpu','logit_normal',torch.Generator().manual_seed(1))
    assert (r<=t).all() and r.min()>=.001 and t.max()<=1
    model=tiny();z=torch.randn(2,4,2,4,6);tokens=torch.randn(2,4,8,384)
    loss,stats=meanflow_loss(model,z,z,tokens,fm_ratio=0,jvp_weight=0)
    assert stats['jvp_backend']=='skipped' and stats['du_dt_rms']==0 and stats['interior_samples']==2
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_independent_clipping_and_legacy_load_and_warm_start(tmp_path):
    a=torch.nn.Linear(1,1);b=torch.nn.Linear(1,1)
    for p in a.parameters():p.grad=torch.ones_like(p)*.01
    for p in b.parameters():p.grad=torch.ones_like(p)*1e6
    original=[p.grad.clone() for p in a.parameters()];stats=clip_training_gradients(a,b)
    assert not stats['dit_clipped'] and stats['policy_clipped']
    for p,expected in zip(a.parameters(),original):torch.testing.assert_close(p.grad,expected)
    model=tiny(1000.);policy=DrivingPolicyHead(hidden_dim=16,depth=1,num_heads=4)
    save_checkpoint(tmp_path,model,policy,500)
    path=tmp_path/'config.json';config=json.loads(path.read_text());del config['dit']['time_scale'];path.write_text(json.dumps(config))
    loaded,_,_=load_checkpoint(tmp_path);assert loaded.config.time_scale==1000.
    z=torch.randn(1,4,2,4,6);tokens=torch.randn(1,4,8,384)
    torch.testing.assert_close(loaded(z,torch.zeros(1),torch.ones(1),z,tokens),model(z,torch.zeros(1),torch.ones(1),z,tokens))
    target=tiny();details=initialize_weights(target,policy,tmp_path)
    assert details==dict(source_step=500,source_time_scale=1000.,target_time_scale=10.)
    for p,q in zip(target.parameters(),model.parameters()):torch.testing.assert_close(p,q)


def test_style_slugs_and_external_teacher_import(tmp_path):
    from unrealityview.data.cache import save_episode
    from unrealityview.data.stylize_cache import import_teacher,StylizedChunkDataset
    from test_style_workflow import FixtureCodec
    assert validate_style_slug('jets_aliens')=='jets_aliens'
    for bad in ('../secret','Upper','', 'x'*33,'a/b'):
        with pytest.raises(ValueError):validate_style_slug(bad)
    base=tmp_path/'base';ep=base/'train/scene'
    save_episode(ep,dict(uve_tokens=torch.randn(64,192,384),chase_latents=torch.zeros(64,4,24,40),ego_states=torch.zeros(64,3)),
        dict(source='pandaset',source_id='fixture/train',frames=64,fps=10,target_view='native_front'))
    raw=ep/'raw/chase';raw.mkdir(parents=True);source=tmp_path/'teacher/train/scene';source.mkdir(parents=True)
    for i in range(64):
        image=Image.new('RGB',(48,32),(i,80,120));image.save(raw/f'{i:05d}.jpg');image.save(source/f'{i:05d}.png')
    digest=hashlib.sha256((ep/'episode.safetensors').read_bytes()).hexdigest()
    metadata=source/'metadata.json';metadata.write_text(json.dumps(dict(frames=64,base_sha256='wrong')))
    with pytest.raises(ValueError,match='fingerprint'):import_teacher(base,tmp_path/'teacher',tmp_path/'out','jets_aliens',decoder=FixtureCodec())
    assert not (tmp_path/'out').exists()
    metadata.write_text(json.dumps(dict(frames=64,base_sha256=digest)))
    import_teacher(base,tmp_path/'teacher',tmp_path/'out','jets_aliens',device='cpu',decoder=FixtureCodec())
    ds=StylizedChunkDataset(base,tmp_path/'out','jets_aliens');assert ds[0]['z0'].shape==(4,8,24,40)
    assert (tmp_path/'out/jets_aliens/train/scene/sanity.png').is_file()
    assert hashlib.sha256((ep/'episode.safetensors').read_bytes()).hexdigest()==digest


def test_matched_phase_network_time_derivative_scale():
    torch.manual_seed(19)
    low=tiny(10.);high=tiny(1000.);high.load_state_dict(low.state_dict())
    z=torch.randn(1,4,2,4,6);v=torch.zeros_like(z);tokens=torch.randn(1,4,8,384);ages=torch.zeros(1,4)
    r=torch.tensor([.1]);t=torch.tensor([.6])
    a=directional_derivative(low,z,r,t,v,z,tokens,ages,'exact')
    b=directional_derivative(high,z,r/100,t/100,v,z,tokens,ages,'exact')
    torch.testing.assert_close(b,a*100,rtol=.003,atol=.0001)


def test_policy_disabled_skips_training_forward_and_warm_start_reset(tmp_path,monkeypatch):
    import train_cached_meanflow as trainer
    from unrealityview.data.cache import save_episode
    from unrealityview.models.lora import model_fingerprint
    data=tmp_path/'data'
    for split in ('train','val'):
        save_episode(data/split/'scene',dict(uve_tokens=torch.randn(64,192,384),
            chase_latents=torch.zeros(64,4,24,40),ego_states=torch.zeros(64,3)),
            dict(frames=64,source_id=f'fixture/{split}'))
    class DisabledPolicy(DrivingPolicyHead):
        def forward(self,*a,**k):raise AssertionError('Disabled policy should not run')
    policy=DisabledPolicy(hidden_dim=16,depth=1,num_heads=4)
    source=tmp_path/'source';model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4))
    save_checkpoint(source,model,policy,500)
    before=model_fingerprint(policy)
    monkeypatch.setattr(trainer,'DrivingPolicyHead',lambda:copy.deepcopy(policy))
    monkeypatch.setattr(trainer,'PretrainedTAESD',torch.nn.Identity)
    monkeypatch.setattr(trainer,'evaluate_models',lambda *a,**k:{'nfe':{'1':{'psnr_db':1.}}})
    config=dict(dit=model.config.to_dict(),steps=1,batch_size=1,grad_accum=1,learning_rate=.001,
        warmup_steps=1,ema_decay=.9,eval_every=1,eval_items=1,seed=1,jvp_method='exact',policy_weight=0.,jvp_warmup_steps=100)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    trainer.train(SimpleNamespace(config=path,steps=None,batch_size=None,device='cpu',data=data,
        output=tmp_path/'out',resume=None,init_weights=source))
    loaded,after,cfg=load_checkpoint(tmp_path/'out/checkpoint')
    assert cfg['step']==1 and model_fingerprint(after)==before
    row=json.loads((tmp_path/'out/train_metrics.jsonl').read_text())
    assert row['policy_grad_norm']==0 and row['policy_loss']==0 and row['jvp_weight']==0


def test_evaluation_baselines_against_raw_jpegs(tmp_path):
    from evaluate import evaluate_models
    from unrealityview.data.cache import save_episode
    from unrealityview.data.dataset import CachedEpisodeChunkDataset
    from test_style_workflow import FixtureCodec
    root=tmp_path/'data';ep=root/'val/scene'
    save_episode(ep,dict(uve_tokens=torch.randn(64,192,384),chase_latents=torch.zeros(64,4,24,40),ego_states=torch.zeros(64,3)),
        dict(source_id='fixture/val',frames=64,fps=10,target_view='native_front'))
    raw=ep/'raw/chase';raw.mkdir(parents=True)
    for i in range(64):Image.new('RGB',(48,32),(128,128,128)).save(raw/f'{i:05d}.jpg')
    model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4))
    policy=DrivingPolicyHead(hidden_dim=16,depth=1,num_heads=4)
    report=evaluate_models(model,policy,FixtureCodec(),CachedEpisodeChunkDataset(root,'val',jitter=False),
        torch.device('cpu'),tmp_path/'evaluation',1)
    assert report['taesd_roundtrip']['pixel_l1']==pytest.approx(1/255,rel=.001)
    assert report['prev_chunk_copy']['latent_l1']==0
    for name in ('closed_loop_32_frames.gif','contact_sheet.png','reconstruction.png'):
        with Image.open(tmp_path/'evaluation'/name) as im:im.verify()
