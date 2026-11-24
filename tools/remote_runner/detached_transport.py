"""Transport helper used by the global vast-run owner, not a separate launcher.

Reconnect to the same remote worker after SSH loss; never resubmit its workload.
The owning runner retains provisioning, caps, retries, artifact sync and cleanup.
"""
import hashlib
from pathlib import Path
import shlex
import subprocess
import time


class DetachedTransportError(RuntimeError):
    pass


def worker_scripts(script,workdir,token,timeout_seconds):
    root=str(Path(workdir)/'.vast-managed'/token);q=shlex.quote(root)
    worker=f'''#!/usr/bin/env bash
set -u
cd {q}
exec 8>worker.lock
flock -n 8 || exit 0
if [ -f exit_code ] || [ -f started ]; then exit 0; fi
echo $$ > worker.pid
touch started
set +e
timeout --signal=TERM --kill-after=60s {int(timeout_seconds)} bash payload.sh >worker.log 2>&1
rc=$?
printf '%s\\n' "$rc" >exit_code.tmp
mv exit_code.tmp exit_code
'''
    launch=f'''set -euo pipefail
mkdir -p {q}
cd {q}
exec 9>launch.lock
flock -x 9
if [ -f exit_code ] || [ -f started ]; then exit 0; fi
printf '%s' {shlex.quote(script)} >payload.sh.tmp
mv payload.sh.tmp payload.sh
printf '%s' {shlex.quote(worker)} >worker.sh.tmp
mv worker.sh.tmp worker.sh
nohup setsid bash worker.sh </dev/null >/dev/null 2>&1 9>&- &
'''
    status=f'''set -euo pipefail
cd {q}
if [ -f exit_code ]; then printf 'EXIT:'; cat exit_code
elif [ -f worker.pid ] && kill -0 "$(cat worker.pid)" 2>/dev/null; then echo RUNNING
elif [ ! -f started ]; then echo STARTING
else echo LOST; fi
'''
    return root,launch,status


def run_detached(ssh_argv,script,workdir,log_path,*,timeout_seconds=21600,
                 outage_seconds=600,poll_seconds=15,execute=subprocess.run,
                 sleep=time.sleep,clock=time.monotonic):
    """Return the remote shell's recorded exit code, including a genuine 255.

    SSH 255 itself is transport failure, never evidence of a workload exit code.
    A launch acknowledgement may be lost; repeated launch is guarded remotely.
    """
    token=hashlib.sha256((script+'\n'+Path(log_path).name).encode()).hexdigest()[:24]
    root,launch,status=worker_scripts(script,workdir,token,timeout_seconds)
    started=clock();last_success=started;launched=False
    def invoke(payload):
        try:
            return execute([*ssh_argv,'bash','-s'],input=payload,text=True,capture_output=True,
                           check=False,timeout=45)
        except (subprocess.TimeoutExpired,OSError):return None
    while clock()-started<timeout_seconds+120:
        result=invoke(status if launched else launch)
        if result is None or result.returncode:
            if result is not None and result.returncode!=255:
                raise DetachedTransportError('Remote launcher/status command failed')
            if clock()-last_success>=outage_seconds:
                raise DetachedTransportError('Remote connection unavailable beyond bounded recovery window')
            sleep(poll_seconds);continue
        last_success=clock()
        if not launched:
            launched=True;sleep(poll_seconds);continue
        state=result.stdout.strip()
        if state.startswith('EXIT:'):
            try:code=int(state.split(':',1)[1])
            except ValueError:raise DetachedTransportError('Invalid recorded remote exit code') from None
            if not 0<=code<=255:raise DetachedTransportError('Remote exit code out of range')
            # Retrieve the full durable log after completion, without streaming
            # per-step output through the model's context during execution.
            logs=invoke(f'cat {shlex.quote(root+"/worker.log")}')
            if logs is None or logs.returncode:
                sleep(poll_seconds);continue
            path=Path(log_path);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as stream:stream.write(logs.stdout)
            return code
        if state=='LOST':raise DetachedTransportError('Remote worker disappeared without a recorded exit code')
        if state not in ('STARTING','RUNNING'):raise DetachedTransportError('Unexpected remote worker state')
        sleep(poll_seconds)
    raise DetachedTransportError('Managed remote workload exceeded its wall-clock deadline')
