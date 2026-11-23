"""Local ZIP fixtures verify parallel transport and restart integrity, without cloud jobs."""
from collections import Counter
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import zipfile
import pytest
from scripts import fetch_pandaset_cameras as fetcher


@pytest.fixture
def archive(tmp_path,monkeypatch):
    path=tmp_path/'source.zip'
    names=[f'pandaset/{i:03d}/camera/front_camera/timestamps.json' for i in range(1,104)]
    selected=fetcher.selected_sequences(names)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        for sequence,_ in selected:
            for camera in fetcher.PANDA_CAMERAS:
                root=f'pandaset/{sequence}/camera/{camera}/'
                z.writestr(root+'intrinsics.json',json.dumps(dict(fx=100,fy=100,cx=20,cy=10)))
                for name in ('poses.json','timestamps.json'):z.writestr(root+name,json.dumps(list(range(80))))
                for index in range(8):z.writestr(root+f'{index:02d}.jpg',bytes([index])*100)
            z.writestr(f'pandaset/{sequence}/lidar/00.pkl',b'not camera data')
    real_open=fetcher.fsspec.open
    monkeypatch.setattr(fetcher.fsspec,'open',lambda url,mode,block_size:real_open(str(path),mode))
    monkeypatch.setattr(fetcher,'HfApi',lambda **k:SimpleNamespace(dataset_info=lambda _:SimpleNamespace(sha='frozen-revision')))
    return path,selected


def test_parallel_fetch_resume_repair_and_frozen_order(archive,tmp_path,monkeypatch):
    _,selected=archive;root=tmp_path/'native';manifest=tmp_path/'manifest.json'
    original=fetcher.download_sequence;lock=threading.Lock();active=0;peak=0;barrier=threading.Barrier(4)
    def tracked(*a,**k):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        try:barrier.wait(timeout=10);return original(*a,**k)
        finally:
            with lock:active-=1
    monkeypatch.setattr(fetcher,'download_sequence',tracked)
    first=fetcher.fetch(root,manifest,max_frames=4,max_bytes=1000000,workers=4)
    assert peak==4 and len(first['download']['members'])==4*6*7
    assert not list(root.rglob('*.pkl')) and not list(root.rglob('04.jpg'))
    assert json.loads(next(root.rglob('timestamps.json')).read_text())==[0,1,2,3]
    assert [e['sequence'] for e in first['sequences']]==[s for s,_ in selected]
    monkeypatch.setattr(fetcher,'download_sequence',original)
    # Reusing an existing session must pin its revision rather than re-query main.
    monkeypatch.setattr(fetcher,'HfApi',lambda **k:pytest.fail('Resume must retain frozen revision'))
    reads=[];read=zipfile.ZipFile.read
    def record(self,member,*a,**k):reads.append(member.filename);return read(self,member,*a,**k)
    monkeypatch.setattr(zipfile.ZipFile,'read',record)
    second=fetcher.fetch(root,manifest,max_frames=4,max_bytes=1000000,workers=2)
    assert second==first and reads==[]
    damaged=next(root.rglob('00.jpg'));damaged.write_bytes(b'broken')
    fetcher.fetch(root,manifest,max_frames=4,max_bytes=1000000,workers=2)
    assert len(reads)==1 and damaged.read_bytes()==bytes([0])*100
    with pytest.raises(ValueError,match='plan changed'):
        fetcher.fetch(root,manifest,max_frames=3,max_bytes=1000000)
    with pytest.raises(ValueError,match='plan changed'):
        fetcher.fetch(root,manifest,max_frames=3,max_bytes=1000000,resume=False)


def test_budget_checked_before_extraction(archive,tmp_path):
    with pytest.raises(ValueError,match='budget'):
        fetcher.fetch(tmp_path/'native',tmp_path/'manifest.json',max_frames=4,max_bytes=1)
    assert not list((tmp_path/'native').rglob('*.jpg'))


def test_crc_failure_does_not_publish_bad_file_or_manifest(archive,tmp_path):
    path,_=archive
    with zipfile.ZipFile(path) as z:
        info=next(i for i in z.infolist() if i.filename.endswith('00.jpg'))
        offset=info.header_offset+30+len(info.filename.encode())+len(info.extra)
        bad_member=Path(info.filename).relative_to('pandaset')
    with path.open('r+b') as f:f.seek(offset);f.write(b'x')
    root=tmp_path/'native';manifest=tmp_path/'manifest.json'
    with pytest.raises(zipfile.BadZipFile,match='CRC'):
        fetcher.fetch(root,manifest,max_frames=4,max_bytes=1000000)
    assert not (root/bad_member).exists() and not manifest.exists()


def test_nonfinite_json_cannot_be_published(archive,tmp_path):
    path,_=archive;replacement=tmp_path/'replacement.zip'
    with zipfile.ZipFile(path) as source,zipfile.ZipFile(replacement,'w') as target:
        changed=False
        for info in source.infolist():
            payload=source.read(info)
            if not changed and info.filename.endswith('timestamps.json'):
                payload=json.dumps([float('nan')]*80).encode();changed=True
            target.writestr(info,payload)
    replacement.replace(path)
    manifest=tmp_path/'manifest.json'
    with pytest.raises(ValueError,match='Out of range float'):
        fetcher.fetch(tmp_path/'native',manifest,max_frames=4,max_bytes=1000000)
    assert not manifest.exists()


def test_preflight_selection_is_pinned_and_recovery_skips_native_fetch(archive,tmp_path,monkeypatch):
    _,selected=archive
    report=dict(revision='frozen-revision',frames=4,seed=42,
                sequences=[dict(dataset='pandaset',sequence=s,split=split) for s,split in selected])
    monkeypatch.setattr(fetcher,'HfApi',lambda **k:pytest.fail('Validated selection must pin revision'))
    from unrealityview.data import cache_recovery
    recovered='pandaset/'+selected[0][0]
    monkeypatch.setattr(cache_recovery,'recoverable_episodes',lambda *a:{recovered:tmp_path/'verified'})
    calls=[];original=fetcher.download_sequence
    def tracked(url,root,sequence,*a,**k):calls.append(sequence);return original(url,root,sequence,*a,**k)
    monkeypatch.setattr(fetcher,'download_sequence',tracked)
    result=fetcher.fetch(tmp_path/'native',tmp_path/'manifest.json',max_frames=4,max_bytes=1000000,
                         selection=report,recovery_cache=tmp_path/'recovery')
    assert selected[0][0] not in calls and len(calls)==3
    assert result['download']['reused_cache_source_ids']==[recovered]
    assert len(result['sequences'])==4 and len(result['download']['members'])==168
    bad={**report,'sequences':[{**e,'sequence':'../secret'} for e in report['sequences']]}
    with pytest.raises(ValueError,match='identity'):
        fetcher.fetch(tmp_path/'other',tmp_path/'bad.json',max_frames=4,selection=bad)
