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
    flag='    run_parser.add_argument("--discard-snapshot-after-upload", action="store_true", help="Discard this regenerable local upload archive after verified transfer")'
    anchor='    run_parser.add_argument("--dry-run-search", action="store_true")'
    if '--discard-snapshot-after-upload' not in source and anchor in source:
        source=source.replace(anchor,anchor+'\n'+flag)
    upload='                upload_snapshot(connection, config.ssh_private_key, paths["snapshot"], remote_workdir)'
    if 'snapshot_retained=False' not in source and upload in source:
        expanded='                if not paths["snapshot"].is_file():\n                    create_repo_snapshot(project_root, paths["snapshot"])\n                upload_snapshot(connection, config.ssh_private_key, paths["snapshot"], remote_workdir)\n                if getattr(args, "discard_snapshot_after_upload", False):\n                    paths["snapshot"].unlink()\n                    update_metadata(paths["metadata"], snapshot_retained=False)'
        source=source.replace(upload,expanded,1)
    compile(source,str(path),'exec')
    shutil.copy2(Path(__file__).with_name('detached_transport.py'),path.with_name('vast_detached.py'))
    path.write_text(source)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runner',required=True);a=p.parse_args();install(a.runner)
