"""Real model/decoder route test without pretrained downloads or a GPU."""
from uuid import uuid4
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from unrealityview.data.cache import save_episode
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.driving_policy_head import DrivingPolicyHead
from unrealityview.models.pretrained_taesd import PretrainedTAESD


@pytest.mark.parametrize("geometry",[False,True])
def test_model_and_ground_truth_routes_and_closed_loop_state(tmp_path,monkeypatch,geometry):
    from app import server
    monkeypatch.setattr(server,"DATA_ROOT",tmp_path)
    directory=tmp_path/"val"/"episode_0000"
    save_episode(directory,{"uve_tokens":torch.randn(32,192,384),"chase_latents":torch.randn(32,4,24,40),
                            "ego_states":torch.zeros(32,3),"intrinsics":torch.eye(3).repeat(32,6,1,1),"camera_to_ego":torch.eye(4).repeat(32,6,1,1)}, {"frames":32,"source":"unit-test fixture"})
    for name in ("front_left","front","front_right","rear_left","rear","rear_right","chase"):
        path=directory/"raw"/name;path.mkdir(parents=True)
        for frame in range(32):Image.new("RGB",(320,192),(40,70,90)).save(path/f"{frame:05d}.jpg")
    model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4,geometry_conditioning=geometry,qk_logit_scale=geometry)).eval()
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
    # A real-data front baseline must preserve native cadence and never apply
    # the elevated-camera raycast or projected chase waypoints to front images.
    from unrealityview.data.cache import atomic_json
    atomic_json(directory/"metadata.json",{"frames":32,"source":"pandaset","fps":10.,"target_view":"native_front"})
    native=client.post("/api/render",json=base)
    assert native.status_code==200
    assert native.json()["fps"]==10 and native.json()["target_view"]=="native_front"
    assert native.json()["projected_waypoints"]==[]
    assert client.post("/api/render",json={**base,"goal":{"u":180.,"v":110.}}).status_code==422
    # Switching styles can re-render the same closed-loop chunk without advancing
    # context. The planner sees identical real tokens through every merge.
    import copy
    from unrealityview.models.lora import inject_dit_lora,save_lora,model_fingerprint,LoRALinear
    adapters=tmp_path/"adapters";adapters.mkdir()
    monkeypatch.setattr(server,"LORA_DIR",adapters)
    adapted=copy.deepcopy(model);inject_dit_lora(adapted,rank=4,start_block=0)
    for style in ("comic","cyberpunk"):
        for layer in adapted.modules():
            if isinstance(layer,LoRALinear):torch.nn.init.normal_(layer.lora_b,std=.03)
        save_lora(adapters/f"{style}_lora.safetensors",adapted,style,1)
    assert client.get("/api/health").json()["styles_available"]==["reality","comic","cyberpunk","hybrid"]
    policy_before=model_fingerprint(policy);base_before=model_fingerprint(model)
    previous=None
    for style in ("reality","comic","cyberpunk","hybrid","reality"):
        response=client.post("/api/render",json={**closed,"chunk":2,"style":style,"style_strength":1.25})
        assert response.status_code==200,response.text
        payload=response.json()
        if previous is None:previous=payload
        assert payload["waypoints"]==previous["waypoints"]
        assert payload["telemetry"]["active_style"]==style
        if style!="reality":assert payload["telemetry"]["lora_size_mb"]>0
    assert payload["frames"]==previous["frames"]
    assert payload["telemetry"]["psnr_reference"]=="real_reference"
    t=payload["telemetry"]
    assert t["effective_fps"]==pytest.approx(8000/(t["dit_ms"]+t["decode_ms"]))
    assert t["request_compute_ms"]>=t["dit_ms"]+t["decode_ms"]
    monkeypatch.setattr(server,"STYLES_DATA_DIR",tmp_path/"teachers")
    import evaluate
    original_metrics=evaluate.pixel_metrics;references=[]
    def capture_reference(prediction,target):
        references.append(target.detach().cpu())
        return original_metrics(prediction,target)
    monkeypatch.setattr(evaluate,"pixel_metrics",capture_reference)
    for style in ("comic","cyberpunk"):
        teacher=server.STYLES_DATA_DIR/style/"val/episode_0000/frames";teacher.mkdir(parents=True)
        for i in range(16,24):Image.new("RGB",(320,192),(130,20,190)).save(teacher/f"{i:05d}.png")
        response=client.post("/api/render",json={**closed,"chunk":2,"style":style})
        assert response.status_code==200,response.text
        assert response.json()["telemetry"]["psnr_reference"]=="style_teacher"
        torch.testing.assert_close(references[-1][0,:,0,0,0],torch.tensor([130.,20.,190.])/127.5-1)
    response=client.post("/api/render",json={**closed,"chunk":2,"style":"hybrid"})
    assert response.json()["telemetry"]["psnr_db"] is None
    assert response.json()["telemetry"]["psnr_reference"]=="unavailable"
    client.post("/api/render",json={**closed,"chunk":2,"style":"reality"})
    assert model_fingerprint(model)==base_before and model_fingerprint(policy)==policy_before
