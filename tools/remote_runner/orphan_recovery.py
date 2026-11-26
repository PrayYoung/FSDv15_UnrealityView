"""Budget watchdog also owns collection if the foreground controller disappears."""
import os,time,subprocess
from types import SimpleNamespace


def wait_for_completion(deadline,path,config,owner,*,probe=os.kill,sleep=time.sleep,now=None,recover_completed=None):
    now=now or owner['utcnow']
    if recover_completed is None:
        from vast_completion_recovery import recover
        recover_completed=lambda run_id:recover(SimpleNamespace(run_id=run_id),config,owner)
    while (remaining:=(deadline-now()).total_seconds())>0:
        metadata=owner['read_json'](path)
        if metadata.get('instance_destroyed_at'):return True
        pid=metadata.get('runner_process',{}).get('pid');dead=False
        if isinstance(pid,int) and pid>0:
            try:probe(pid,0)
            except ProcessLookupError:dead=True
            except PermissionError:pass
        if dead and metadata.get('ssh_host') and metadata.get('remote_workdir'):
            try:
                recover_completed(metadata['run_id'])
            except (RuntimeError,BlockingIOError,TimeoutError,OSError,subprocess.SubprocessError):
                # Workload may still be active or another recovery may own its
                # lock. Never restart it; the hard deadline remains in force.
                pass
            else:return True
        sleep(min(15.,remaining))
    return False
