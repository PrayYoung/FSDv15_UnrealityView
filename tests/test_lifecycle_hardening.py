import io,gzip,tarfile,subprocess,os
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tools.remote_runner.lifecycle_runtime import skip_login_banner,wait_wall_deadline
from tools.remote_runner.lifecycle_install import patch


def test_binary_archive_survives_login_banner():
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz') as archive:
        info=tarfile.TarInfo('metrics.json');payload=b'{"ok":true}';info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
    with tarfile.open(fileobj=skip_login_banner(io.BytesIO(b'Welcome to Vast\n'+raw.getvalue())),mode='r|gz') as archive:
        assert archive.extractfile(next(iter(archive))).read()==payload


def test_deadline_rechecks_wall_time_after_suspend():
    start=datetime.now(timezone.utc);times=iter([start,start+timedelta(hours=5)]);sleeps=[]
    wait_wall_deadline(start+timedelta(hours=1),now=lambda:next(times),sleep=sleeps.append)
    assert sleeps==[5.]


def test_installer_matches_current_owner_and_is_idempotent():
    source=(Path.home()/'vast-run/vast_run.py').read_text();after=patch(source)
    assert patch(after)==after
    assert 'transfer_timeout_seconds=120' in after and 'cleanup_verified_absent=true' in after
    assert 'wait_wall_deadline(deadline)' in after and 'shell_command = f"sleep' not in after


def test_hook_import_survives_upstream_tools_package(tmp_path):
    from tools.chase_3dgs.setup_upstream import instrument_optimizer_progress
    (tmp_path/'tools').mkdir();(tmp_path/'tools/__init__.py').write_text('')
    source=tmp_path/'train.py';source.write_text('def train(trainer, step, loss_dict):\n        trainer.backward(loss_dict)\ntrain(type("Trainer",(),{"backward":lambda self,loss:None})(),9,{"rgb":.1})\n')
    instrument_optimizer_progress(source)
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'PYTHONPATH':str(root/'src'),'UNREALITY_PROGRESS_FILE':str(tmp_path/'progress.json')}
    subprocess.run([__import__('sys').executable,str(source)],cwd=tmp_path,env=env,check=True)
    assert '"completed_updates": 10' in (tmp_path/'progress.json').read_text()
