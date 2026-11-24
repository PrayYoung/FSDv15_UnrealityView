from types import SimpleNamespace
import subprocess
import pytest
from tools.remote_runner.detached_transport import run_detached,worker_scripts,DetachedTransportError


def test_disconnect_during_launch_and_wait_does_not_repeat_workload(tmp_path):
    calls=[];responses=iter([(255,''),(0,''),(255,''),(0,'RUNNING'),(0,'EXIT:0'),(0,'saved remote log')])
    def execute(argv,**kwargs):
        calls.append(kwargs['input']);code,out=next(responses);return SimpleNamespace(returncode=code,stdout=out)
    code=run_detached(['ssh','host'],'echo workload','/remote/project',tmp_path/'log',
                      execute=execute,sleep=lambda _:None,clock=lambda:0)
    assert code==0 and (tmp_path/'log').read_text()=='saved remote log'
    assert calls[0]==calls[1] and 'nohup setsid' in calls[0]
    assert calls[2]==calls[3] and 'nohup' not in calls[2]
    _,launch,_=worker_scripts('echo task','/remote/project','id',60)
    assert 'flock -x' in launch and '[ -f started ]' in launch


def test_real_workload_255_is_returned_only_from_exit_file(tmp_path):
    replies=iter([(0,''),(0,'EXIT:255'),(0,'python failed')])
    def execute(*a,**k):code,out=next(replies);return SimpleNamespace(returncode=code,stdout=out)
    assert run_detached(['ssh'],'body','/job',tmp_path/'log',execute=execute,sleep=lambda _:None,clock=lambda:0)==255


def test_bounded_transport_failure_and_lost_worker(tmp_path):
    times=iter([0,0,601]);execute=lambda *a,**k:SimpleNamespace(returncode=255,stdout='')
    with pytest.raises(DetachedTransportError,match='recovery window'):
        run_detached(['ssh'],'body','/job',tmp_path/'log',execute=execute,sleep=lambda _:None,clock=lambda:next(times))
    responses=iter(['','LOST'])
    with pytest.raises(DetachedTransportError,match='disappeared'):
        run_detached(['ssh'],'body','/job',tmp_path/'log',execute=lambda *a,**k:SimpleNamespace(returncode=0,stdout=next(responses)),sleep=lambda _:None,clock=lambda:0)


def test_installer_preserves_lifecycle_and_is_idempotent(tmp_path):
    from tools.remote_runner.install import install
    path=tmp_path/'vast_run.py'
    path.write_text('''def ssh_base_command():
    return ["ServerAliveCountMax=4",]

def upload_snapshot():
    pass

remote_exit_code = ssh_run_script(
                    connection,
                    config.ssh_private_key,
                    remote_script,
                    log_path=paths["remote_log"],
                )
''')
    install(path);first=path.read_text();install(path)
    assert path.read_text()==first and 'TCPKeepAlive=yes' in first
    assert path.with_suffix('.before-detached.py').exists()
    assert path.with_name('vast_detached.py').exists()


def test_worker_never_starts_fails_promptly(tmp_path):
    responses=iter(['','STARTING']);times=iter([0,0,0,0,0,61,61])
    with pytest.raises(DetachedTransportError,match='startup grace'):
        run_detached(['ssh'],'body','/job',tmp_path/'log',execute=lambda *a,**k:SimpleNamespace(returncode=0,stdout=next(responses)),sleep=lambda _:None,clock=lambda:next(times))


def test_snapshot_disposal_is_opt_in_and_retry_rebuilds(tmp_path):
    from tools.remote_runner.install import install
    import runpy
    from pathlib import Path
    path=tmp_path/'runner.py'
    path.write_text('''from pathlib import Path
uploaded=[]
updates=[]
def build_parser():
    run_parser=None
    run_parser.add_argument("--dry-run-search", action="store_true")
def upload_snapshot(connection, key, snapshot, remote_workdir):
    uploaded.append(snapshot.read_bytes())
def create_repo_snapshot(project_root,snapshot):
    snapshot.write_bytes(b'rebuilt')
def update_metadata(path,**kwargs):updates.append(kwargs)
def unused():
    remote_exit_code = ssh_run_workload()
def simulate(args,project_root,paths,config):
    connection=None
    remote_workdir='/remote'
    for once in [0]:
        if True:
            if True:
                upload_snapshot(connection, config.ssh_private_key, paths["snapshot"], remote_workdir)
''')
    install(path);module=runpy.run_path(str(path));archive=tmp_path/'snapshot.tar.gz';archive.write_bytes(b'original')
    metadata=tmp_path/'metadata.json';metadata.write_text('{}');outputs=tmp_path/'weights.safetensors';outputs.write_bytes(b'unique')
    paths=dict(snapshot=archive,metadata=metadata);config=SimpleNamespace(ssh_private_key=Path('/key'))
    module['simulate'](SimpleNamespace(discard_snapshot_after_upload=False),tmp_path,paths,config)
    assert archive.exists()
    module['simulate'](SimpleNamespace(discard_snapshot_after_upload=True),tmp_path,paths,config)
    assert not archive.exists()
    module['simulate'](SimpleNamespace(discard_snapshot_after_upload=True),tmp_path,paths,config)
    assert module['uploaded']==[b'original',b'original',b'rebuilt']
    assert outputs.read_bytes()==b'unique' and module['updates'][-1]['snapshot_retained'] is False
