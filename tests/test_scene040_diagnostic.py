import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.chase_3dgs.diagnostic_common import FRAMES,SAMPLES,SIZE,chase_pose,preview,seal_views
from tools.chase_3dgs.native_scene import NativeScene
from tools.chase_3dgs.extract_pandaset_scene import CAMERAS
from unrealityview.data.real_world import CV_TO_FLU


def test_diagnostic_pose_geometry_and_discrete_orbit_keep_pivot_radius():
    front=np.eye(4);front[:3,:3]=CV_TO_FLU
    a,k=chase_pose(front,'A');b,kb=chase_pose(front,'B')
    np.testing.assert_allclose(a[:3,3],[-6.8,0,3.8]);np.testing.assert_allclose(b[:3,3],[-5.5,0,3.])
    assert SIZE==(640,384) and len(FRAMES)==68 and SAMPLES==[12,25,40,55,70]
    for yaw in (-25,0,25):
        pose,intrinsic=chase_pose(front,'B',yaw=yaw)
        np.testing.assert_allclose(np.linalg.norm(pose[:2,3]),5.5)
        assert pose[2,3]==3;np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6)
        np.testing.assert_allclose(intrinsic,kb)


def test_native_reader_uses_exact_pose_timestamps_and_scaled_intrinsics(tmp_path):
    pose=dict(heading=dict(w=1.,x=0.,y=0.,z=0.),position=dict(x=10.,y=20.,z=30.))
    times=[i/10 for i in range(80)]
    for name in CAMERAS:
        p=tmp_path/'040/camera'/name;p.mkdir(parents=True)
        (p/'intrinsics.json').write_text(json.dumps(dict(fx=100.,fy=80.,cx=4.,cy=3.)))
        (p/'poses.json').write_text(json.dumps([pose]*80));(p/'timestamps.json').write_text(json.dumps(times))
        Image.new('RGB',(8,6),(40,50,60)).save(p/'12.jpg')
    p=tmp_path/'040/lidar';p.mkdir();(p/'timestamps.json').write_text(json.dumps(times))
    scene=NativeScene(tmp_path);rgb,k,poses,offsets=scene.cameras(12,size=(4,3))
    assert rgb.shape==(6,3,4,3) and offsets==[0.]*6
    np.testing.assert_allclose(k[0],[[50,0,2],[0,40,1.5],[0,0,1]])
    np.testing.assert_allclose(poses[0,:3,3],[10,20,30])
    assert scene.front(12,SIZE).size==SIZE
    (p/'timestamps.json').unlink()
    camera_only=NativeScene(tmp_path,use_lidar=False)
    assert camera_only.cameras(12,size=(4,3))[0].shape==(6,3,4,3)
    import pytest
    with pytest.raises(RuntimeError,match='disabled'):camera_only.lidar_world(12)


def test_previews_do_not_stretch_and_exact_frame_range_is_recorded(tmp_path):
    image=Image.new('RGB',SIZE,'white');small=preview(image,(512,320))
    assert small.getpixel((256,0))==(0,0,0) and small.getpixel((256,160))==(255,255,255)
    seal_views(tmp_path,{'pose_A':{i:f'pose_A/{i}.png' for i in FRAMES}},{i:f'ref/{i}.png' for i in FRAMES},list(range(80)))
    result=json.loads((tmp_path/'views.json').read_text());assert result['timestamps']==list(range(12,80))
