import os
from pathlib import Path
import threading
import time
from uuid import uuid4
import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient
from unrealityview.checkpoint import save_checkpoint,load_checkpoint
from unrealityview.data.cache import resolve_safe_subpath,save_episode
from unrealityview.data.dataset import CachedEpisodeChunkDataset
from unrealityview.geometry import ego_waypoints,plucker_rays,ground_raycast,project_waypoints
from unrealityview.models.mini_uve import MiniUVE
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.driving_policy_head import DrivingPolicyHead,training_goals
from unrealityview.models.pretrained_taesd import PretrainedTAESD
from unrealityview.objectives import meanflow_loss,directional_derivative
from unrealityview.runtime.ring_buffer_engine import UVERingBuffer,RealityViewStreamingEngine


def tiny_model():
    return ConditionalMeanFlowDiT(DiTConfig(hidden_dim=32,depth=4,num_heads=4,latent_shape=(2,4,6)))


def inputs(batch=2):
    return (torch.randn(batch,4,2,4,6),torch.randn(batch,4,2,4,6),torch.randn(batch,4,192,384))


def test_plucker_geometry_and_mini_uve():
    k=torch.tensor([[100.,0,112],[0,100,70],[0,0,1]]).expand(2,6,3,3)
    ext=torch.eye(4).expand(2,6,4,4).clone();ext[...,0,3]=1
    rays=plucker_rays(k,ext)
    assert rays.shape==(2,6,32,6)
    torch.testing.assert_close(rays[...,:3].norm(dim=-1),torch.ones(2,6,32))
    assert (rays[...,:3]*rays[...,3:]).sum(-1).abs().max()<1e-6
    class Backbone(nn.Module):
        num_prefix_tokens=1
        def forward_features(self,x):return torch.ones(len(x),161,384)
    model=MiniUVE(backbone=Backbone())
    cameras=torch.rand(2,6,3,140,224)
    assert model(cameras,k,ext).shape==(2,192,384)
    assert not model(cameras,k,ext).requires_grad
    with pytest.raises(ValueError):model(cameras,k)


@pytest.mark.parametrize("steps",[1,2,4])
def test_sampling_shape_and_previous_channel_concat(steps):
    model=tiny_model();z,prev,tokens=inputs()
    stages=[]
    output=model.sample_chunk(prev,tokens,steps,noise=z,uve_provider=lambda stage:stages.append(stage))
    assert output.shape==prev.shape
    assert torch.isfinite(output).all()
    assert stages==[0,1,2,3]*steps
    assert model.position.shape[1]==6
    assert model.patch.in_channels==8
    with pytest.raises(ValueError):model.sample_chunk(prev,tokens,3)


def test_meanflow_exact_jvp_matches_finite_and_has_finite_gradients():
    torch.manual_seed(9)
    model=tiny_model()
    nn.init.normal_(model.output.weight,std=.03)
    for block in model.blocks:nn.init.normal_(block.mod[-1].weight,std=.03)
    z,prev,tokens=inputs()
    r,t=torch.tensor([.1,.2]),torch.tensor([.8,.7])
    v=torch.randn_like(z);ages=torch.zeros(2,4)
    exact=directional_derivative(model,z,r,t,v,prev,tokens,ages,"exact")
    finite=directional_derivative(model,z,r,t,v,prev,tokens,ages,"finite",epsilon=1e-4)
    torch.testing.assert_close(exact,finite,atol=.005,rtol=.02)
    for method in ("exact","finite"):
        loss,stats=meanflow_loss(model,z,prev,tokens,ages,fm_ratio=0,jvp_method=method)
        assert stats["interior_samples"]==2
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)


def test_flow_matching_boundary_skips_jvp(monkeypatch):
    import unrealityview.objectives as objectives
    monkeypatch.setattr(objectives,"directional_derivative",lambda *args,**kwargs:pytest.fail("boundary must skip JVP"))
    z,prev,tokens=inputs()
    loss,stats=meanflow_loss(tiny_model(),z,prev,tokens,fm_ratio=1)
    assert stats["interior_samples"]==0 and torch.isfinite(loss)


def test_taesd_tiled_equivalence_and_sram_budget():
    decoder=PretrainedTAESD(pretrained=False,precise=True)
    z=torch.randn(1,4,1,4,6)
    full=decoder.decode(z)
    tiled=decoder.decode_tiled(z,latent_tile=2,latent_halo=8)
    assert full.shape==(1,3,1,32,48)
    torch.testing.assert_close(full,tiled,atol=2e-6,rtol=2e-5)
    budget=decoder.sram_budget_report()
    assert budget["within_budget"] and budget["total_mib"]<32
    assert budget["int8_weight_bytes"]>1_000_000


def test_policy_goal_conditioning_dropout_and_independence():
    torch.manual_seed(4)
    policy=DrivingPolicyHead(hidden_dim=32,num_heads=4,depth=1)
    tokens=torch.randn(3,192,384)
    left=policy(tokens,torch.tensor([[30.,5.]]).expand(3,2))
    right=policy(tokens,torch.tensor([[30.,-5.]]).expand(3,2))
    assert left.shape==(3,12,4)
    assert (left-right).abs().max()>1e-4
    torch.testing.assert_close(left[...,2:].norm(dim=-1),torch.ones(3,12))
    waypoints=torch.ones(2000,12,4)
    goals=training_goals(waypoints)
    zeros=(goals==0).all(1).float().mean()
    assert .45<float(zeros)<.55


def test_ring_snapshot_is_immutable_and_survives_slot_reuse():
    ring=UVERingBuffer(2)
    original=torch.ones(1,2,3)
    first=ring.publish(original,ts=0)
    original.zero_()
    for i in range(5):ring.publish(torch.full((1,2,3),float(i)),ts=i)
    assert first.tokens.sum()==6
    assert ring.read_latest_nonblocking().sequence_id==5


def test_multithread_interleaving_freshness_gain():
    ring=UVERingBuffer();ready=threading.Event();advance=threading.Event();done=threading.Event()
    def producer():
        ring.publish(torch.zeros(1,192,384),ts=1,frame_id=0);ready.set()
        assert advance.wait(5)
        ring.publish(torch.ones(1,192,384),ts=1.1,frame_id=4);done.set()
    thread=threading.Thread(target=producer);thread.start();assert ready.wait(5)
    class Consumer:
        def sample_chunk(self,previous,tokens,num_steps,uve_provider,noise):
            for stage in range(4):
                if stage==1:advance.set();assert done.wait(5)
                uve_provider(stage)
            return previous
    engine=RealityViewStreamingEngine(Consumer(),ring)
    engine.sample_chunk(torch.zeros(1,4,2,4,6),clock=lambda:1.2)
    thread.join(5)
    assert [s["frame_id"] for s in engine.last_telemetry["stages"]]==[0,4,4,4]
    assert engine.last_telemetry["freshness_gain_ms"]==pytest.approx(100)
    assert not thread.is_alive()


def test_ego_rotation_and_raycast_roundtrip():
    poses=torch.tensor([[0.,0.,torch.pi/2],[0.,1.,torch.pi/2],[0.,2.,torch.pi/2]])
    wp=ego_waypoints(poses,0,2,1)
    torch.testing.assert_close(wp[:,:2],torch.tensor([[1.,0.],[2.,0.]]),atol=1e-6,rtol=0)
    for goal in ([10.,0.],[30.,-3.],[12.,4.]):
        u,v=project_waypoints([goal])[0]
        assert ground_raycast(u,v)==pytest.approx(goal)
    assert ground_raycast(160,0) is None


def test_safe_path_rejects_traversal_absolute_and_symlinks(tmp_path):
    root=tmp_path/"data";root.mkdir()
    for path in ("../secret","/etc/passwd","x/../../secret"):
        with pytest.raises(ValueError):resolve_safe_subpath(root,path)
    (root/"escape").symlink_to(tmp_path)
    with pytest.raises(ValueError):resolve_safe_subpath(root,"escape/file")


def test_cached_chunks_jitter_is_causal_and_split_isolated(tmp_path):
    ep={"uve_tokens":torch.randn(80,192,384),"chase_latents":torch.randn(80,4,24,40),
        "ego_states":torch.zeros(80,3)}
    save_episode(tmp_path/"train"/"ep",ep,{"frames":80})
    data=CachedEpisodeChunkDataset(tmp_path)
    item=data[0]
    assert item["z0"].shape==(4,8,24,40)
    assert item["uve_tokens"].shape==(4,192,384)
    assert (item["stage_ages"]>=0).all()
    with pytest.raises(ValueError):CachedEpisodeChunkDataset(tmp_path,"val")


def test_safetensors_checkpoint_roundtrip(tmp_path):
    model=tiny_model();policy=DrivingPolicyHead(hidden_dim=32,num_heads=4,depth=1)
    save_checkpoint(tmp_path,model,policy,7)
    loaded,p2,config=load_checkpoint(tmp_path)
    assert config["step"]==7
    for a,b in zip(model.parameters(),loaded.parameters()):torch.testing.assert_close(a,b)
    for a,b in zip(policy.parameters(),p2.parameters()):torch.testing.assert_close(a,b)


def test_api_health_render_validation_and_local_guards(monkeypatch,tmp_path):
    from app import server
    monkeypatch.setattr(server,"DATA_ROOT",tmp_path)
    client=TestClient(server.app)
    assert client.get("/").status_code==200
    assert client.get("/api/health").json()["status"]=="ok"
    base={"session_id":str(uuid4()),"source":"ground_truth","episode":"val/recording"}
    response=client.post("/api/render",json=base)
    assert response.status_code==404  # No synthetic frames replace missing recordings.
    for bad in ({"nfe":3},{"nfe":True},{"nfe":1.0},{"chunk":-1},{"chunk":"1"},{"tiled":"true"},
                {"goal":{"u":999.,"v":100.}},{"unknown":1},{"source":"demo"},
                {"style":"unknown"},{"style_strength":-0.1},{"style_strength":1.6}):
        assert client.post("/api/render",json={**base,**bad}).status_code==422
    assert client.post("/api/render",json={**base,"episode":"../secret"}).status_code==400
    assert client.post("/api/render",json=base,headers={"Origin":"https://evil.example"}).status_code==403
    assert client.get("/api/health",headers={"Host":"evil.example"}).status_code==400



@pytest.mark.pretrained
@pytest.mark.skipif(os.environ.get("RUN_PRETRAINED")!="1",reason="Opt-in downloaded pretrained weights")
def test_actual_pretrained_vision_and_taesd():
    from unrealityview.device import select_device
    device=select_device()
    encoder=MiniUVE().to(device)
    assert encoder(torch.rand(1,6,3,140,224,device=device)).shape==(1,192,384)
    decoder=PretrainedTAESD(precise=True).to(device)
    z=torch.randn(1,4,1,24,40,device=device)
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        full,tiled=decoder.decode(z),decoder.decode_tiled(z)
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:torch.backends.cuda.matmul.allow_tf32=previous
    torch.testing.assert_close(full,tiled,atol=5e-5,rtol=1e-4)
    assert decoder.sram_budget_report()["within_budget"]
