import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from unrealityview.data.pandaset_preflight import validate_camera_metadata
from unrealityview.data.real_world import PANDA_CAMERAS
from unrealityview.data.cache import save_episode,atomic_json
from unrealityview.data.cache_recovery import recoverable_episodes,seed_recovery,RAW_CAMERAS


def metadata():
    return {c:dict(intrinsics=dict(fx=100.,fy=100.,cx=20.,cy=10.),timestamps=(np.arange(80)/10).tolist(),
        image_indices=list(range(80)),poses=[dict(heading=dict(w=1.,x=0.,y=0.,z=0.),position=dict(x=0.,y=0.,z=0.)) for _ in range(80)]) for c in PANDA_CAMERAS}


def test_metadata_preflight_rejects_duplicate_clock_without_repair():
    cameras=metadata();assert validate_camera_metadata(cameras)['fps']==pytest.approx(10)
    cameras['back_camera']['timestamps'][67]=cameras['back_camera']['timestamps'][66]
    original=copy.deepcopy(cameras)
    with pytest.raises(ValueError,match='back_camera.*strictly increasing'):validate_camera_metadata(cameras)
    assert cameras==original


@pytest.mark.parametrize('problem',['skew','gap','pose','images','intrinsics'])
def test_metadata_preflight_covers_reader_failures(problem):
    cameras=metadata();camera=cameras['right_camera']
    if problem=='skew':camera['timestamps']=[t+2 for t in camera['timestamps']]
    if problem=='gap':camera['timestamps'][40:]=[t+.5 for t in camera['timestamps'][40:]]
    if problem=='pose':camera['poses'][0]['heading']['w']=0
    if problem=='images':camera['image_indices'].pop()
    if problem=='intrinsics':camera['intrinsics']['fx']=float('nan')
    with pytest.raises(ValueError):validate_camera_metadata(cameras)


def test_verified_recovery_skips_native_data_but_preserves_tensors(tmp_path,monkeypatch):
    from prepare_real_data import cache_manifest
    old=dict(seed=42,download=dict(revision='frozen'),sequences=[dict(dataset='pandaset',sequence='002',split='train',root='old-native')])
    digest=hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()
    recovery=tmp_path/'recovery';atomic_json(recovery/'split_manifest.json',old)
    name='pandaset_'+hashlib.sha256(b'pandaset/002').hexdigest()[:16];directory=recovery/'train'/name
    save_episode(directory,dict(uve_tokens=torch.zeros(80,192,384,dtype=torch.float16),chase_latents=torch.zeros(80,4,24,40,dtype=torch.float16),ego_states=torch.zeros(80,3)),
        dict(source='pandaset',source_id='pandaset/002',split='train',frames=80,target_view='native_front',input_size=[224,140],pixel_size=[320,192],split_manifest_sha256=digest))
    for camera in RAW_CAMERAS:
        path=directory/'raw'/camera;path.mkdir(parents=True)
        for i in range(80):(path/f'{i:05d}.jpg').write_bytes(b'fixture')
    new=dict(seed=42,download=dict(revision='frozen',frames=80),sequences=[dict(dataset='pandaset',sequence='002',split='train',root='missing-native')])
    expected=hashlib.sha256((directory/'episode.safetensors').read_bytes()).hexdigest()
    assert len(recoverable_episodes(recovery,'frozen',new['sequences'],80))==1
    output=tmp_path/'new';assert seed_recovery(recovery,output,new)==1
    import prepare_real_data
    monkeypatch.setattr(prepare_real_data,'PandaSetSequence',lambda *a:pytest.fail('Must not reopen native data for a verified recovery'))
    cache_manifest(new,output,'cpu',encoder=torch.nn.Identity(),decoder=torch.nn.Identity(),reuse_existing=True)
    assert hashlib.sha256((output/'train'/name/'episode.safetensors').read_bytes()).hexdigest()==expected
    assert json.loads((directory/'metadata.json').read_text())['split_manifest_sha256']==digest
    with pytest.raises(ValueError,match='revision'):recoverable_episodes(recovery,'wrong',new['sequences'],80)
    (directory/'raw/chase/00000.jpg').unlink()
    with pytest.raises(ValueError,match='incomplete'):recoverable_episodes(recovery,'frozen',new['sequences'],80)
