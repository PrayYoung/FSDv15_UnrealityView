"""Recover only a completed existing Vast job after its original owner died."""
import fcntl
import os
from pathlib import Path
import re
import shlex
import subprocess


def assert_owner_absent(metadata,probe=os.kill):
    pid=metadata.get('runner_process',{}).get('pid')
    if not isinstance(pid,int) or pid<=0:raise RuntimeError('Missing original owner PID')
    try:probe(pid,0)
    except ProcessLookupError:return
    raise RuntimeError('Original local owner is still alive; refusing duplicate recovery')


def recorded_exit(stdout):
    matches=re.findall(r'^UNREALITY_EXIT=(\d+)\s*$',stdout,re.MULTILINE)
    if len(matches)!=1 or not 0<=int(matches[0])<=255:
        raise RuntimeError('Existing workload has no unique recorded terminal exit; not restarting or terminating it')
    return int(matches[0])



def terminal_script(workdir):
    return f'''set -euo pipefail
cd {shlex.quote(workdir)}
terminal=0
for file in .vast-managed/*/exit_code; do
 if [ -f "$file" ]; then printf 'UNREALITY_EXIT='; cat "$file"; terminal=1; fi
done
if [ "$terminal" = 1 ]; then
 for file in .vast-managed/*/worker.log; do
  if [ -f "$file" ]; then tail -c 65536 "$file"; fi
 done
fi
'''


def recover(args,config,owner):
    path,metadata=owner['load_metadata_by_run_id'](config,args.run_id)
    assert_owner_absent(metadata)
    if metadata.get('instance_destroyed_at'):raise RuntimeError('Instance already destroyed; inspect local artifacts')
    from vast_lifecycle import prevent_idle_sleep
    prevent_idle_sleep()
    with Path(path).with_suffix('.recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        connection=owner['SshConnection'](host=metadata['ssh_host'],port=int(metadata['ssh_port']))
        script=terminal_script(metadata['remote_workdir'])
        result=subprocess.run([*owner['ssh_base_command'](connection,config.ssh_private_key),'bash','-s'],
            input=script,text=True,capture_output=True,check=False,timeout=30)
        if result.returncode:raise RuntimeError('Cannot read existing terminal receipt: '+result.stderr[-300:])
        code=recorded_exit(result.stdout)
        Path(metadata['remote_log_path']).write_text(result.stdout)
        update=owner['update_metadata'];update(path,recovery_started_at=owner['utcnow_iso'](),remote_exit_code=code,recovery_owner_pid=os.getpid())
        classification='SUCCESS' if code==0 else 'WORKLOAD_FAILED'
        try:
            owner['emit_phase'](path,'SYNCING','recover completed existing job; no resubmission')
            paths=owner['sync_remote_paths'](connection,config.ssh_private_key,metadata['remote_workdir'],
                tuple(metadata['sync_paths']),Path(metadata['outputs_dir']),transfer_timeout_seconds=300)
            update(path,synced_paths=paths,finished_at=owner['utcnow_iso'](),final_classification=classification)
        except Exception as error:
            classification='ARTIFACT_FAILED';update(path,final_classification=classification,recovery_artifact_error=str(error))
        finally:
            owner['emit_phase'](path,'CLEANING_UP')
            try:
                owner['destroy_instance'](int(metadata['instance_id']))
                owner['confirm_instance_gone'](int(metadata['instance_id']))
                update(path,status='DESTROYED',instance_destroyed_at=owner['utcnow_iso'](),cleanup_pending=False,cleanup_initiated_by='completed_job_recovery')
            except Exception as error:
                update(path,cleanup_pending=True,cleanup_error=str(error),final_classification='INFRASTRUCTURE_FAILED')
                owner['schedule_watchdog'](config,args.run_id,owner['utcnow'](),Path(path).parent/'cleanup_watchdog.log')
                raise
        final=owner['finalize_summary_fields'](path);owner['print_terminal_summary'](final)
        return 0 if classification=='SUCCESS' else 1
