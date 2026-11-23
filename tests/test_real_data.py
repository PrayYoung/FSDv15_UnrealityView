"""Native-format fixtures verify adapters; they are never called real recordings."""
import json
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
pytest.importorskip("pyarrow")
from fsd_reality.data.real_world import PandaSetSequence,WaymoSequence,PANDA_CAMERAS,CV_TO_FLU,resize_calibrated
from fsd_reality.data.splits import stable_split,validate_split_manifest,audit_cache_splits,assert_unseen_test
from fsd_reality.data.cache import atomic_json
from fsd_reality.data.dataset import CachedEpisodeChunkDataset
from fsd_reality.data.dataset import stage_context
from fsd_reality.models.mini_uve import MiniUVE
from fsd_reality.models.blocks import Attention
from prepare_real_data import cache_manifest


def panda_fixture(root,sequence,frames=48):
    for name in PANDA_CAMERAS:
        path=root/sequence/"camera"/name;path.mkdir(parents=True)
        atomic_json(path/"intrinsics.json",dict(fx=20.,fy=20.,cx=16.,cy=10.))
        atomic_json(path/"timestamps.json",[100.+i/10 for i in range(frames)])
        atomic_json(path/"poses.json",[dict(heading=dict(w=.5,x=-.5,y=.5,z=-.5),
                                           position=dict(x=i*.2,y=0.,z=1.5)) for i in range(frames)])
        for i in range(frames):Image.new("RGB",(32,20),(30+i,70,110)).save(path/f"{i:02d}.jpg")


def waymo_fixture(root,sequence,frames=3):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from io import BytesIO
    buffer=BytesIO();Image.new("RGB",(32,20),(40,80,100)).save(buffer,"JPEG")
    image=buffer.getvalue();calibration=[];poses=[];images=[]
    for camera in range(1,6):
        row={"key.camera_name":camera,"[CameraCalibrationComponent].extrinsic.transform":np.eye(4).flatten().tolist()}
        for key,value in dict(f_u=20.,f_v=20.,c_u=16.,c_v=10.,k1=0.,k2=0.,p1=0.,p2=0.,k3=0.).items():
            row["[CameraCalibrationComponent].intrinsic."+key]=value
        calibration.append(row)
    for i in range(frames):
        stamp=100000000+i*100000;matrix=np.eye(4);matrix[0,3]=i*.2
        poses.append({"key.frame_timestamp_micros":stamp,"[VehiclePoseComponent].world_from_vehicle.transform":matrix.flatten().tolist()})
        for camera in range(1,6):
            images.append({"key.frame_timestamp_micros":stamp,"key.camera_name":camera,
                           "[CameraImageComponent].image":image,"[CameraImageComponent].pose.transform":matrix.flatten().tolist(),
                           "[CameraImageComponent].pose_timestamp":stamp/1e6})
    for component,rows in (("camera_calibration",calibration),("vehicle_pose",poses),("camera_image",images)):
        path=root/component;path.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows),path/(sequence+".parquet"),row_group_size=7)


def test_pandaset_camera_only_geometry_and_timing(tmp_path):
    panda_fixture(tmp_path,"001",3)
    sequence=PandaSetSequence(tmp_path,"001")
    frame=sequence.frame(1)
    assert len(frame.images)==6 and frame.camera_mask.all()
    np.testing.assert_allclose(frame.ego_pose,[.2,0,0],atol=1e-7)
    np.testing.assert_allclose(frame.camera_to_ego[1,:3,:3],CV_TO_FLU,atol=1e-7)
    assert frame.timestamp==pytest.approx(100.1)
    assert frame.images[0].size==(224,140)
    assert frame.intrinsics[0,0,0]==pytest.approx(140.)
    # No lidar directory or pickle is needed or created.
    assert not (tmp_path/"001/lidar").exists()


def test_waymo_five_cameras_rectification_and_pose_join(tmp_path):
    waymo_fixture(tmp_path,"context")
    frame=WaymoSequence(tmp_path,"context").frame(1)
    assert frame.camera_mask.tolist()==[True,True,True,True,False,True]
    np.testing.assert_allclose(frame.camera_to_ego[1,:3,:3],CV_TO_FLU)
    np.testing.assert_allclose(frame.ego_pose,[.2,0,0])
    assert np.asarray(frame.images[4]).sum()==0


def test_panda_skew_rejected_and_calibration_scaled(tmp_path):
    panda_fixture(tmp_path,"001",3)
    path=tmp_path/"001/camera/left_camera/timestamps.json"
    atomic_json(path,[101.,101.1,101.2])
    with pytest.raises(ValueError,match="skew"):PandaSetSequence(tmp_path,"001").frame(0)
    image,k=resize_calibrated(Image.new("RGB",(100,50)),np.array([[50.,0,50],[0,50,25],[0,0,1]]),(200,100))
    assert image.size==(200,100)
    np.testing.assert_allclose(k,[[100,0,100],[0,100,50],[0,0,1]])


def test_scene_split_stability_and_leakage(tmp_path):
    assert stable_split("pandaset","001")==stable_split("pandaset","001")
    first=dict(dataset="pandaset",sequence="001",root="unused",split="train")
    with pytest.raises(ValueError,match="more than once"):
        validate_split_manifest({"sequences":[first,{**first,"split":"test"}]})
    for split in ("train","test"):atomic_json(tmp_path/split/"episode/metadata.json",{"source_id":"pandaset/001"})
    with pytest.raises(ValueError,match="leakage"):audit_cache_splits(tmp_path)
    with pytest.raises(ValueError,match="overlap"):assert_unseen_test({"train_source_ids":["pandaset/001"]},["pandaset/001"])
    with pytest.raises(ValueError,match="provenance"):assert_unseen_test(None,[])


def test_native_cache_timestamps_and_frozen_splits(tmp_path):
    root=tmp_path/"native"
    for name in ("001","002","003"):panda_fixture(root,name)
    entries=[dict(dataset="pandaset",sequence=name,root=str(root),split=split)
             for name,split in zip(("001","002","003"),("train","val","test"))]
    class Encoder:
        def __call__(self,cameras,k,ext,camera_mask):return torch.ones(1,192,384)*camera_mask.repeat_interleave(32,dim=1)[:,:,None]
    class Decoder:
        def encode(self,video):return torch.zeros(1,4,1,24,40)
    manifest=dict(version=1,source="native-format unit fixture",sequences=entries)
    output=tmp_path/"cache"
    cache_manifest(manifest,output,"cpu",encoder=Encoder(),decoder=Decoder())
    data=CachedEpisodeChunkDataset(output,jitter=False)
    assert data[0]["fps"]==pytest.approx(10)
    torch.testing.assert_close(data[0]["stage_ages"],torch.full((4,),.1))
    assert set(data.provenance.values())=={"train","val","test"}
    assert data.metadata[0]["target_view"]=="native_front"
    with pytest.raises(ValueError,match="different frozen"):
        cache_manifest({**manifest,"seed":99},output,"cpu",encoder=Encoder(),decoder=Decoder())


def test_missing_camera_zero_tokens_and_attention_mask():
    class Backbone(nn.Module):
        num_prefix_tokens=1
        def forward_features(self,x):return torch.ones(len(x),161,384)
    encoder=MiniUVE(backbone=Backbone())
    mask=torch.tensor([[1,1,1,1,0,1]],dtype=torch.bool)
    tokens=encoder(torch.rand(1,6,3,140,224),camera_mask=mask)
    assert tokens[:,128:160].abs().sum()==0
    attn=Attention(16,4,384)
    x=torch.randn(1,4,16);valid=mask.repeat_interleave(32,1)
    before=attn(x,tokens,key_mask=valid)
    changed=tokens.clone();changed[:,128:160]=10000
    torch.testing.assert_close(before,attn(x,changed,key_mask=valid))
    with pytest.raises(ValueError):encoder(torch.rand(1,6,3,140,224),camera_mask=torch.zeros_like(mask))


def test_camera_bundle_completion_prevents_future_peeking():
    times=torch.arange(30,dtype=torch.float64)/10
    episode={"uve_tokens":torch.zeros(30,192,384),"timestamps":times,
             "camera_timestamps":times[:,None].expand(-1,6).clone()}
    episode["camera_timestamps"][:,3]+=.05
    ids,ages=stage_context(episode,8,fps=10,jitter=torch.ones(4,dtype=torch.long))
    assert ids.tolist()==[7,9,11,13]
    torch.testing.assert_close(ages,torch.full((4,),.05))
    assert (episode["camera_timestamps"][ids].max(1).values<=times[8+torch.tensor([0,2,4,6])]).all()
