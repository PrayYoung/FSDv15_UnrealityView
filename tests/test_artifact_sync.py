import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest
from scripts.seal_return import seal
from tools.remote_runner import artifact_sync, lifecycle_runtime
from tools.remote_runner.artifact_install import patch


@pytest.fixture(autouse=True)
def lifecycle_import(monkeypatch):
    monkeypatch.setitem(sys.modules, 'vast_lifecycle', lifecycle_runtime)


def test_real_archive_roundtrip_recovers_failed_probe_and_verifies_checkpoint(tmp_path):
    remote = tmp_path / 'remote'
    out = remote / 'outputs/return'
    out.mkdir(parents=True)
    (out / 'checkpoint.bin').write_bytes(bytes(range(256)) * 4096)
    (out / 'summary.json').write_text('{"completed_updates":2000}')
    seal(out)
    attempts = []

    def probe(argv, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            return SimpleNamespace(returncode=255, stderr='transport lost')
        return subprocess.run(['bash', '-s'], **kwargs)

    def transfer(argv, **kwargs):
        return subprocess.Popen(argv[1:], env={**os.environ,'COPYFILE_DISABLE':'1'}, **kwargs)

    local = tmp_path / 'local'
    assert artifact_sync.sync_paths(['fake-ssh'], str(remote), ['outputs/return'], local,
        execute=probe, popen=transfer, sleep=lambda _: None) == ['outputs/return']
    assert len(attempts) == 2
    assert (local/'outputs/return/checkpoint.bin').read_bytes() == (out/'checkpoint.bin').read_bytes()
    assert not list(local.glob('.artifact-*'))


def test_probe_loss_never_reports_empty_success(tmp_path):
    def missing(*args, **kwargs):
        return SimpleNamespace(returncode=255, stderr='SSH unavailable')
    with pytest.raises(artifact_sync.ArtifactError, match='attempt 3.*probe exit 255'):
        artifact_sync.sync_paths(['ssh'], '/remote', ['outputs/return'], tmp_path,
            execute=missing, sleep=lambda _: None)


def test_corrupt_archive_does_not_overwrite_existing_checkpoint(tmp_path):
    remote = tmp_path/'remote'; out=remote/'outputs/return';out.mkdir(parents=True)
    (out/'checkpoint.bin').write_bytes(b'original');seal(out)
    (out/'checkpoint.bin').write_bytes(b'corrupt')
    local=tmp_path/'local';old=local/'outputs/return/checkpoint.bin';old.parent.mkdir(parents=True);old.write_bytes(b'preserved')
    with pytest.raises(artifact_sync.ArtifactError, match='checksum mismatch'):
        artifact_sync.sync_paths(['fake'],str(remote),['outputs/return'],local, attempts=1,
            execute=lambda argv, **kwargs: subprocess.run(['bash','-s'],**kwargs),
            popen=lambda argv, **kwargs: subprocess.Popen(argv[1:],env={**os.environ,'COPYFILE_DISABLE':'1'},**kwargs))
    assert old.read_bytes()==b'preserved'


def test_empty_return_and_path_traversal_are_rejected(tmp_path):
    with pytest.raises(artifact_sync.ArtifactError,match='empty'):
        artifact_sync.validate_return(tmp_path)
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz') as archive:
        item=tarfile.TarInfo('../escape');item.size=1;archive.addfile(item,io.BytesIO(b'x'))
    with pytest.raises(artifact_sync.ArtifactError,match='Unsafe'):
        artifact_sync.extract_checked(io.BytesIO(raw.getvalue()),tmp_path)


def test_installer_classifies_artifact_failure_and_is_idempotent():
    source=(Path.home()/'vast-run/vast_run.py').read_text()
    after=patch(source)
    assert patch(after)==after
    assert 'final_classification="ARTIFACT_FAILED"' in after
    assert 'artifact_sync_error=str(error)' in after
    # Run the real injected helper with a local transport: creation, tar retrieval,
    # and exact receipt check, with no workload/GPU launch.
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);remote=root/'remote';remote.mkdir()
        start=after.index('def artifact_transfer_preflight(');end=after.index('def sync_remote_paths(',start)
        def retrieve(connection,key,workdir,paths,destination,transfer_timeout_seconds):
            return artifact_sync.sync_paths(['fake'],workdir,paths,destination,
                execute=lambda argv, **kwargs: subprocess.run(['bash','-s'],**kwargs),
                popen=lambda argv, **kwargs: subprocess.Popen(argv[1:],env={**os.environ,'COPYFILE_DISABLE':'1'},**kwargs))
        import shlex
        namespace=dict(Path=Path,shlex=shlex,subprocess=subprocess,InfrastructureError=RuntimeError,
            ssh_base_command=lambda connection,key: [],
            sync_remote_paths=retrieve)
        exec(after[start:end],namespace)
        namespace['artifact_transfer_preflight'](None,None,str(remote),root/'local')
        assert len(list((root/'local/transfer_preflight').rglob('receipt.txt')))==1


def test_difix_avoids_forward_compat_shim_without_dropping_host_libraries():
    script='source scripts/difix_cuda_env.sh; printf "%s" "$LD_LIBRARY_PATH"'
    result=subprocess.run(['bash','-c',script],env={'LD_LIBRARY_PATH':'/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda-12.4/compat/:/opt/lib'},capture_output=True,text=True,check=True)
    assert result.stdout=='/usr/local/nvidia/lib64:/opt/lib'
