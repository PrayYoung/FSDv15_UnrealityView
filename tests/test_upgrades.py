import copy
import json
import numpy as np
import pytest
import torch
from unrealityview.data.cache import save_episode
from unrealityview.data.dataset import CachedEpisodeChunkDataset
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.pretrained_taesd import PretrainedTAESD
from unrealityview.models.lora import inject_dit_lora,LoRALinear,model_fingerprint
from unrealityview.training import (ema_decay,RestartableSampler,save_training_state,load_training_state,rng_state,restore_rng)


def test_training_stride_and_exact_policy_alignment(tmp_path):
    for split in ('train','val','test'):
        save_episode(tmp_path/split/'scene',{'uve_tokens':torch.arange(80).float()[:,None,None].expand(80,192,384),
            'chase_latents':torch.zeros(80,4,24,40),'ego_states':torch.zeros(80,3)},
            {'frames':80,'source_id':f'fixture/{split}'})
    train=CachedEpisodeChunkDataset(tmp_path,jitter=False)
    val=CachedEpisodeChunkDataset(tmp_path,'val');test=CachedEpisodeChunkDataset(tmp_path,'test')
    assert len(train)==18 and len(val)==len(test)==5
    assert [s for _,s in train.index]==list(range(8,44,2))
    assert train[1]['policy_tokens'][0,0]==10 and train[1]['uve_tokens'][0,0,0]==9
    assert len(CachedEpisodeChunkDataset(tmp_path,stride=8))==5
    with pytest.raises(ValueError):CachedEpisodeChunkDataset(tmp_path,stride=0)


def test_every_block_cross_attention_freshness_and_missing_mask():
    model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=32,depth=4,num_heads=4,latent_shape=(2,4,6)))
    torch.nn.init.normal_(model.output.weight,std=.02)
    for block in model.blocks:torch.nn.init.normal_(block.mod[-1].weight,std=.02)
    calls=[];inputs=[]
    for i,block in enumerate(model.blocks):
        block.cross_attn.register_forward_pre_hook(lambda module,args,i=i:calls.append(i))
        block.cross_attn.kv.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().clone()))
    z=torch.randn(1,4,2,4,6);tokens=torch.randn(1,4,192,384);tokens[:,:,160:]=0
    out=model(z,torch.zeros(1),torch.ones(1),z,tokens,torch.zeros(1,4))
    assert calls==[0,1,2,3] and all(t[:,160:].count_nonzero()==0 for t in inputs)
    assert not torch.equal(out,model(z,torch.zeros(1),torch.ones(1),z,tokens,torch.ones(1,4)))
    assert model.blocks[0].mod[-1].out_features==9*32
    inject_dit_lora(model,rank=8,early_rank=4,start_block=0)
    assert model.blocks[0].cross_attn.q.rank==4 and isinstance(model.output,LoRALinear)


def test_safe_resume_optimizer_ema_rng_and_sampler(tmp_path):
    torch.manual_seed(9);np.random.seed(9)
    model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    ema=copy.deepcopy(model);sampler=RestartableSampler(10,9);stream=iter(sampler)
    def step():
        batch=torch.randn(2,3)+np.random.random();optimizer.zero_grad();model(batch).square().mean().backward();optimizer.step()
        with torch.no_grad():
            for old,new in zip(ema.parameters(),model.parameters()):old.lerp_(new,1-ema_decay(.995,2))
    step();assert next(stream)>=0
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),ema=ema.state_dict(),rng=rng_state(),sampler=sampler.state_dict())
    save_training_state(tmp_path,state);step();expected=model_fingerprint(model);expected_ema=model_fingerprint(ema);next_id=next(stream)
    loaded=load_training_state(tmp_path);model.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer']);ema.load_state_dict(loaded['ema'])
    restore_rng(loaded['rng']);sampler.load_state_dict(loaded['sampler']);step()
    assert model_fingerprint(model)==expected and model_fingerprint(ema)==expected_ema
    assert next(iter(sampler))==next_id
    assert ema_decay(.995,1)==pytest.approx(2/11) and ema_decay(.995,10000)==.995
    meta=json.loads((tmp_path/'training_state.json').read_text());(tmp_path/meta['tensors']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='fingerprint'):load_training_state(tmp_path)


def test_batched_taesd_matches_framewise_and_uses_one_call():
    decoder=PretrainedTAESD(pretrained=False,precise=True);z=torch.randn(2,4,2,4,6)
    calls=[];hook=decoder.vae.decoder.register_forward_hook(lambda *args:calls.append(1))
    batch=decoder.decode(z);hook.remove()
    assert len(calls)==1
    separate=torch.stack([torch.cat([decoder.decode(z[b:b+1,:,t:t+1]) for t in range(2)],2)[0] for b in range(2)])
    torch.testing.assert_close(batch,separate,atol=3e-6,rtol=3e-5)
    pixels=torch.randn(2,3,2,32,48).clamp(-1,1);calls=[]
    hook=decoder.vae.encoder.register_forward_hook(lambda *args:calls.append(1));decoder.encode(pixels);hook.remove()
    assert len(calls)==1 and not PretrainedTAESD(pretrained=False).precise


def test_scaled_fetcher_keeps_frozen_scene_assignments():
    from scripts.fetch_pandaset_cameras import selected_sequences
    from unrealityview.data.splits import stable_split
    names=[f'pandaset/{i:03d}/camera/front_camera/timestamps.json' for i in range(1,104)]
    subset=selected_sequences(names,train_count=50,val_count=4,test_count=4)
    assert len(subset)==58 and len({s for s,_ in subset})==58
    whole=selected_sequences(names,all_sequences=True)
    assert len(whole)==103 and all(stable_split('pandaset',s)==split for s,split in whole)
    assert set(subset)<=set(whole)


def test_teacher_chase_geometry_and_triangulation():
    from prepare_chase_teacher import chase_calibration,triangulate_matches
    from unrealityview.data.real_world import CV_TO_FLU
    from unrealityview.geometry import project_waypoints
    front=np.eye(4);front[:3,:3]=CV_TO_FLU
    pose,k=chase_calibration(front);point=np.array([20.,3.,0.,1.])
    cv=np.linalg.inv(pose)@point;pixel=k@cv[:3];pixel=pixel[:2]/pixel[2]
    np.testing.assert_allclose(pixel,project_waypoints([[20.,3.]])[0],atol=1e-6)
    pose1=np.eye(4);pose2=np.eye(4);pose2[0,3]=1
    points=np.array([[0,0,5],[1,1,8],[-1,.5,10]],np.float32)
    intr=np.array([[100,0,24],[0,100,16],[0,0,1.]])
    def project(pose):
        camera=(np.linalg.inv(pose)@np.c_[points,np.ones(3)].T).T[:,:3];p=(intr@camera.T).T
        return (p[:,:2]/p[:,2:]).astype(np.float32)
    xyz,valid=triangulate_matches(project(pose1),project(pose2),intr,intr,pose1,pose2)
    assert valid.all();np.testing.assert_allclose(xyz,points,atol=1e-4)


def test_base_trainer_resume_matches_uninterrupted_and_keeps_best(tmp_path,monkeypatch):
    import train_cached_meanflow as trainer
    from types import SimpleNamespace
    from unrealityview.checkpoint import load_checkpoint
    from unrealityview.models.driving_policy_head import DrivingPolicyHead
    data=tmp_path/'data'
    for split in ('train','val'):
        save_episode(data/split/'scene',{'uve_tokens':torch.randn(64,192,384),
            'chase_latents':torch.zeros(64,4,24,40),'ego_states':torch.zeros(64,3)},
            {'frames':64,'source_id':f'fixture/{split}'})
    config=dict(dit=dict(hidden_dim=16,depth=4,num_heads=4),steps=2,batch_size=1,grad_accum=1,
        learning_rate=.001,warmup_steps=1,ema_decay=.995,eval_every=1,eval_items=1,seed=7,jvp_method='auto',policy_weight=1.)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    monkeypatch.setattr(trainer,'DrivingPolicyHead',lambda:DrivingPolicyHead(hidden_dim=16,depth=1,num_heads=4))
    monkeypatch.setattr(trainer,'PretrainedTAESD',torch.nn.Identity)
    evaluated=[]
    def evaluate_fixture(*a,**k):
        evaluated.append(a[5].name);return {'nfe':{'1':{'psnr_db':10.}}}
    monkeypatch.setattr(trainer,'evaluate_models',evaluate_fixture)
    args=SimpleNamespace(config=path,steps=None,batch_size=None,device='cpu',output=tmp_path/'full',data=data,evaluate_initial=True)
    trainer.train(args)
    expected,policy,_=load_checkpoint(args.output/'checkpoint')
    args.output=tmp_path/'resumed';args.stop_after=1;trainer.train(args)
    args.resume=args.output/'checkpoint';args.stop_after=None;trainer.train(args)
    actual,actual_policy,config=load_checkpoint(args.output/'checkpoint')
    assert config['step']==2 and model_fingerprint(actual)==model_fingerprint(expected)
    assert model_fingerprint(actual_policy)==model_fingerprint(policy)
    best=json.loads((args.output/'checkpoint_best/config.json').read_text())
    assert best['step']==1 and best['target_views']==['elevated_chase']
    assert evaluated.count('eval_000000')==2  # Fresh runs only; never rerun baseline on resume.


def test_server_cli_paths_and_configured_origin(tmp_path,monkeypatch):
    import sys
    import uvicorn
    from fastapi.testclient import TestClient
    from app import server
    for key in ('DATA_ROOT','CHECKPOINT','LORA_DIR','STYLES_DATA_DIR','PORT'):
        monkeypatch.setattr(server,key,getattr(server,key))
    for key in ('DATA','CHECKPOINT','LORA_DIR','STYLES_DATA','PORT'):
        monkeypatch.setenv('UNREALITYVIEW_'+key,'unchanged-test-value')
    monkeypatch.setattr(sys,'argv',['server','--data',str(tmp_path/'data'),'--checkpoint',str(tmp_path/'weights'),
        '--lora-dir',str(tmp_path/'lora'),'--styles-data-dir',str(tmp_path/'styles'),'--port','8093'])
    calls=[];monkeypatch.setattr(uvicorn,'run',lambda *a,**k:calls.append((a,k)))
    server.main()
    assert calls[0][1]['host']=='127.0.0.1' and calls[0][1]['port']==8093
    assert server.DATA_ROOT==tmp_path/'data' and server.CHECKPOINT==tmp_path/'weights'
    client=TestClient(server.app)
    assert client.get('/api/health',headers={'Origin':'http://localhost:8093'}).status_code==200
    assert client.get('/api/health',headers={'Origin':'http://localhost:8088'}).status_code==403
    assert client.get('/api/health',headers={'Host':'evil.example'}).status_code==400


def test_legacy_checkpoint_loads_original_architecture(tmp_path):
    from unrealityview.checkpoint import save_checkpoint,load_checkpoint
    from unrealityview.models.driving_policy_head import DrivingPolicyHead
    model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4,latent_shape=(2,4,6),cross_attention='stage'))
    policy=DrivingPolicyHead(hidden_dim=16,depth=1,num_heads=4)
    save_checkpoint(tmp_path,model,policy,30)
    path=tmp_path/'config.json';config=json.loads(path.read_text());del config['dit']['cross_attention'];path.write_text(json.dumps(config))
    loaded,_,_=load_checkpoint(tmp_path)
    assert loaded.config.cross_attention=='stage' and hasattr(loaded,'cross')
    assert model_fingerprint(model)==model_fingerprint(loaded)


def test_precise_backend_restores_both_tf32_flags_after_error():
    from unrealityview.models.pretrained_taesd import reference_convolutions
    original=torch.backends.cuda.matmul.allow_tf32
    enabled=torch.backends.cudnn.enabled;tf32=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        with pytest.raises(RuntimeError,match='probe'):
            with reference_convolutions():
                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.enabled and not torch.backends.cudnn.allow_tf32
                raise RuntimeError('probe')
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.enabled==enabled and torch.backends.cudnn.allow_tf32==tf32
    finally:torch.backends.cuda.matmul.allow_tf32=original


def test_decoder_respects_explicit_autocast_context():
    from types import SimpleNamespace
    from unrealityview.models.pretrained_taesd import precise_convolutions
    @precise_convolutions
    def probe(self,tensor):
        return torch.is_autocast_enabled(tensor.device.type),torch.get_autocast_dtype(tensor.device.type),tensor.dtype
    tensor=torch.ones(1)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        assert probe(SimpleNamespace(precise=False),tensor)[:2]==(True,torch.bfloat16)
        enabled,_,dtype=probe(SimpleNamespace(precise=True),tensor.half())
        assert not enabled and dtype==torch.float32
