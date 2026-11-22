"""Real model/decoder route test without pretrained downloads or a GPU."""
from uuid import uuid4
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from fsd_reality.data.cache import save_episode
from fsd_reality.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from fsd_reality.models.driving_policy_head import DrivingPolicyHead
from fsd_reality.models.pretrained_taesd import PretrainedTAESD


def test_model_and_ground_truth_routes_and_closed_loop_state(tmp_path,monkeypatch):
    from app import server
    monkeypatch.setattr(server,"DATA_ROOT",tmp_path)
    directory=tmp_path/"val"/"episode_0000"
    save_episode(directory,{"uve_tokens":torch.randn(32,192,384),"chase_latents":torch.randn(32,4,24,40),
                            "ego_states":torch.zeros(32,3)}, {"frames":32,"source":"unit-test fixture"})
    for name in ("front_left","front","front_right","rear_left","rear","rear_right","chase"):
        path=directory/"raw"/name;path.mkdir(parents=True)
        for frame in range(32):Image.new("RGB",(320,192),(40,70,90)).save(path/f"{frame:05d}.jpg")
    model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4)).eval()
    policy=DrivingPolicyHead(hidden_dim=16,depth=1,num_heads=4).eval()
    decoder=PretrainedTAESD(pretrained=False)
    monkeypatch.setattr(server,"_models",(model,policy,decoder,torch.device("cpu"),{"step":1,"data_source":"unit-test fixture"}))
    monkeypatch.setattr(server,"_episode_cache",server.OrderedDict())
    monkeypatch.setattr(server,"_sessions",server.OrderedDict())
    client=TestClient(server.app)
    base={"session_id":str(uuid4()),"source":"ground_truth","episode":"val/episode_0000"}
    gt=client.post("/api/render",json=base)
    assert gt.status_code==200
    assert client.get(gt.json()["frames"][0]).status_code==200
    closed={**base,"source":"model","rollout":"closed_loop"}
    assert client.post("/api/render",json={**closed,"chunk":2}).status_code==409
    first=client.post("/api/render",json=closed)
    assert first.status_code==200,first.text
    data=first.json()
    assert data["trained"] is True and data["training_data_source"]=="unit-test fixture"
    assert data["telemetry"]["dit_ms"]>0 and data["telemetry"]["policy_ms"]>0
    assert data["telemetry"]["uve_stage_frames"]==[7,9,11,13]
    second=client.post("/api/render",json={**closed,"chunk":2})
    assert second.status_code==200,second.text
    assert client.post("/api/render",json={**closed,"chunk":3,"nfe":2}).status_code==409
    assert client.post("/api/render",json={**base,"chunk":5}).status_code==422
