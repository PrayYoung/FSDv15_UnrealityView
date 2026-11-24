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
