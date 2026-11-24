"""Install detached transport into the existing vast-run lifecycle owner.

Explicit --runner path; back up its source before changing it. No cloud calls.
"""
import argparse
from pathlib import Path
import shutil


def install(path):
    path=Path(path).resolve();source=path.read_text();backup=path.with_suffix('.before-detached.py')
    if not backup.exists():shutil.copy2(path,backup)
    source=source.replace('"ServerAliveCountMax=4",','"ServerAliveCountMax=10",\n        "-o",\n        "TCPKeepAlive=yes",')
    helper='''def ssh_run_workload(connection, ssh_private_key_path, script, *, remote_workdir, log_path):
    from vast_detached import run_detached, DetachedTransportError
    try:
        return run_detached(ssh_base_command(connection, ssh_private_key_path), script,
                            remote_workdir, log_path)
    except DetachedTransportError as error:
        raise InfrastructureError(str(error)) from error


'''
    if 'def ssh_run_workload(' not in source:
        if 'def upload_snapshot(' not in source:raise ValueError('Unsupported runner source layout')
        source=source.replace('def upload_snapshot(',helper+'def upload_snapshot(',1)
    old='''remote_exit_code = ssh_run_script(
                    connection,
                    config.ssh_private_key,
                    remote_script,
                    log_path=paths["remote_log"],
                )'''
    new=old.replace('ssh_run_script(', 'ssh_run_workload(').replace('log_path=paths["remote_log"],','remote_workdir=remote_workdir,\n                    log_path=paths["remote_log"],')
    source=source.replace(old,new)
    if 'remote_exit_code = ssh_run_workload(' not in source:raise ValueError('Unsupported runner workload entry point')
    source=source.replace('exit_code = ssh_run_script(connection, config.ssh_private_key, remote_script, log_path=rerun_log, interactive=False)',
        'exit_code = ssh_run_workload(connection, config.ssh_private_key, remote_script, remote_workdir=remote_workdir, log_path=rerun_log)')
    compile(source,str(path),'exec')
    shutil.copy2(Path(__file__).with_name('detached_transport.py'),path.with_name('vast_detached.py'))
    path.write_text(source)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runner',required=True);a=p.parse_args();install(a.runner)
