import gzip
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.chase_3dgs.extract_pandaset_scene import extract,selected_members,validate_native_scene,CAMERAS


def archive_spec(path):
    return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_gzip_native_tables_are_sdk_discoverable_and_complete(tmp_path,monkeypatch):
    monkeypatch.setattr('tools.chase_3dgs.extract_pandaset_scene.shutil.disk_usage',lambda _:SimpleNamespace(free=100*2**30))
    archive=tmp_path/'native.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for relative in [f'camera/{c}' for c in CAMERAS]+['lidar']:
            for name in ('poses.json','timestamps.json'):
                z.writestr(f'pandaset/040/{relative}/{name}',json.dumps([0,1]))
        for i in range(2):
            for c in CAMERAS:z.writestr(f'pandaset/040/camera/{c}/{i:02d}.jpg',b'image')
            for relative in ('lidar','annotations/cuboids'):
                z.writestr(f'pandaset/040/{relative}/{i:02d}.pkl.gz',gzip.compress(b'opaque table'))
    root=tmp_path/'out';record=extract(archive,root,['040'],archive_spec(archive))
    assert record['bytes']>0 and not list(root.rglob('*.pkl.gz'))
    assert (root/'040/lidar/00.pkl').read_bytes()==b'opaque table'
    assert len(validate_native_scene(root,'040',expected_frames=2))==8
    (root/'040/lidar/01.pkl').unlink()
    with pytest.raises(ValueError,match='Incomplete native frames'):validate_native_scene(root,'040',expected_frames=2)


def test_gzip_expansion_limit_and_target_collision(tmp_path,monkeypatch):
    monkeypatch.setattr('tools.chase_3dgs.extract_pandaset_scene.shutil.disk_usage',lambda _:SimpleNamespace(free=100*2**30))
    archive=tmp_path/'native.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('pandaset/040/lidar/00.pkl.gz',gzip.compress(b'x'*10000))
    with pytest.raises(ValueError,match='Expanded'):extract(archive,tmp_path/'out',['040'],archive_spec(archive),max_bytes=1000)
    assert not list((tmp_path/'out').rglob('*.part')) and not list((tmp_path/'out').rglob('*.pkl'))
    with zipfile.ZipFile(archive,'a') as z:z.writestr('pandaset/040/lidar/00.pkl',b'duplicate')
    with zipfile.ZipFile(archive) as z:
        with pytest.raises(ValueError,match='Duplicate native extraction target'):selected_members(z,'040')
