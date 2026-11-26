from pathlib import Path
import pytest
from tools.remote_runner.completion_recovery import assert_owner_absent,recorded_exit
from tools.remote_runner.completion_install import patch


def test_recovery_refuses_live_or_unknown_owner():
    with pytest.raises(RuntimeError,match='still alive'):assert_owner_absent({'runner_process':{'pid':123}},probe=lambda *a:None)
    with pytest.raises(RuntimeError,match='Missing'):assert_owner_absent({})
    def gone(*a):raise ProcessLookupError()
    assert_owner_absent({'runner_process':{'pid':123}},probe=gone)


def test_only_existing_terminal_receipt_is_accepted():
    assert recorded_exit('Welcome\nUNREALITY_EXIT=0\nlog')==0
    assert recorded_exit('UNREALITY_EXIT=124\n')==124
    for text in ('RUNNING','UNREALITY_EXIT=0\nUNREALITY_EXIT=1\n','UNREALITY_EXIT=999\n'):
        with pytest.raises(RuntimeError):recorded_exit(text)


def test_installer_is_idempotent_and_never_submits_a_job():
    source=(Path.home()/'vast-run/vast_run.py').read_text();after=patch(source)
    assert patch(after)==after and 'from vast_completion_recovery import recover' in after
    import inspect
    from tools.remote_runner.completion_recovery import recover
    body=inspect.getsource(recover)
    assert 'create_instance' not in body and 'assert_owner_absent(metadata)' in body
    assert 'transfer_timeout_seconds=300' in body


def test_watchdog_recovers_dead_owner_but_never_live_owner():
    from datetime import datetime,timedelta,timezone
    from tools.remote_runner.orphan_recovery import wait_for_completion
    start=datetime.now(timezone.utc);metadata={'run_id':'existing','runner_process':{'pid':1},'ssh_host':'host','remote_workdir':'/job'}
    owner={'read_json':lambda p:metadata};actions=[];times=iter([start,start+timedelta(seconds=2)])
    assert not wait_for_completion(start+timedelta(seconds=1),'metadata',None,owner,probe=lambda *a:None,
        sleep=lambda _:None,now=lambda:next(times),recover_completed=actions.append)
    assert not actions
    def gone(*a):raise ProcessLookupError()
    assert wait_for_completion(start+timedelta(seconds=1),'metadata',None,owner,probe=gone,
        sleep=lambda _:None,now=lambda:start,recover_completed=actions.append)
    assert actions==['existing']
    metadata['instance_destroyed_at']='closed';actions.clear()
    assert wait_for_completion(start+timedelta(seconds=1),'metadata',None,owner,probe=gone,
        sleep=lambda _:None,now=lambda:start,recover_completed=actions.append)
    assert not actions


@pytest.mark.parametrize('error',[RuntimeError('No terminal receipt'),__import__('subprocess').TimeoutExpired('ssh',30)])
def test_watchdog_does_not_reset_budget_for_unfinished_orphan(error):
    from datetime import datetime,timedelta,timezone
    from tools.remote_runner.orphan_recovery import wait_for_completion
    start=datetime.now(timezone.utc);times=iter([start,start+timedelta(seconds=2)]);calls=[]
    def not_done(run):calls.append(run);raise error
    def gone(*a):raise ProcessLookupError()
    owner={'read_json':lambda p:{'run_id':'existing','runner_process':{'pid':1},'ssh_host':'host','remote_workdir':'/job'}}
    assert not wait_for_completion(start+timedelta(seconds=1),'metadata',None,owner,probe=gone,
        sleep=lambda _:None,now=lambda:next(times),recover_completed=not_done)
    assert calls==['existing']


def test_unfinished_job_does_not_download_log_tail(tmp_path):
    import subprocess
    from tools.remote_runner.completion_recovery import terminal_script
    work=tmp_path/'.vast-managed/one';work.mkdir(parents=True)
    (work/'worker.log').write_text('diagnostic log must wait for completion')
    script=terminal_script(str(tmp_path))
    result=subprocess.run(['bash','-s'],input=script,text=True,capture_output=True,check=True)
    assert result.stdout==''
    (work/'exit_code').write_text('0\n')
    result=subprocess.run(['bash','-s'],input=script,text=True,capture_output=True,check=True)
    assert recorded_exit(result.stdout)==0 and 'diagnostic log' in result.stdout
