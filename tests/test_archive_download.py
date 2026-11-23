import hashlib
import json
from types import SimpleNamespace
import httpx
import pytest
from scripts import download_pandaset_archive as bulk
from scripts import fetch_pandaset_cameras as fetcher
from test_fetch_resume import archive


def test_verified_download_resume_and_reject_corruption(tmp_path, monkeypatch):
    payload=b'public fixture';spec=dict(repo='example/public',filename='data.zip',revision='a'*40,
        bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(bulk.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*2**30))
    calls=[];waits=[]
    def client(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            response=httpx.Response(429,headers={'Retry-After':'90'},request=httpx.Request('GET','https://example.com'))
            raise httpx.HTTPStatusError('limited',request=response.request,response=response)
        target=tmp_path/'data.zip';target.write_bytes(payload);return str(target)
    path=bulk.download(spec,tmp_path,tmp_path/'record.json',client,waits.append)
    assert len(calls)==2 and waits==[90] and calls[0]['revision']==spec['revision']
    bulk.download(spec,tmp_path,tmp_path/'record.json',lambda **_:pytest.fail('must reuse complete file'))
    assert json.loads((tmp_path/'record.json').read_text())['verified_sha256']==spec['sha256']
    path.write_bytes(b'x'*len(payload))
    with pytest.raises(ValueError,match='SHA256'):bulk.verify_archive(path,spec)


def test_permanent_error_and_disk_guard(tmp_path,monkeypatch):
    for status in (401,403,404):
        response=httpx.Response(status,request=httpx.Request('GET','https://example.com'))
        error=httpx.HTTPStatusError('permanent',request=response.request,response=response)
        assert bulk.retry_delay(error,0) is None
    assert bulk.retry_delay(httpx.ConnectTimeout('timeout'),0)==30
    monkeypatch.setattr(bulk.shutil,'disk_usage',lambda _:SimpleNamespace(free=1))
    with pytest.raises(OSError,match='free space'):
        bulk.download(dict(filename='data.zip',bytes=100),tmp_path,tmp_path/'record.json')


def test_local_archive_extraction_without_network(archive,tmp_path,monkeypatch):
    path,selected=archive
    # Undo fixture transport redirect: real local fsspec transport is required.
    monkeypatch.undo()
    selection=dict(revision='frozen-revision',frames=4,seed=42,
        sequences=[dict(dataset='pandaset',sequence=s,split=split) for s,split in selected])
    monkeypatch.setattr(fetcher,'HfApi',lambda **_:pytest.fail('no network resolution'))
    manifest=fetcher.fetch(tmp_path/'native',tmp_path/'manifest.json',max_frames=4,max_bytes=1000000,
        selection=selection,local_archive=path)
    assert len(manifest['download']['members'])==168
    assert manifest['download']['transport'].startswith('verified local ZIP')
    assert not list((tmp_path/'native').rglob('*.pkl'))
