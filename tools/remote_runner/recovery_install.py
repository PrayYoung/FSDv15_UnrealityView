"""Install opt-in host exclusion plus bounded infrastructure recovery in vast-run."""
import argparse
from pathlib import Path
import shutil


def patch(source):
    flag='    run_parser.add_argument("--exclude-host-id", action="append", type=int, default=None, help="Exclude a known failed host from this job")'
    anchor='    run_parser.add_argument("--dry-run-search", action="store_true")'
    if '--exclude-host-id' not in source:
        if anchor not in source:raise ValueError('Unsupported parser')
        source=source.replace(anchor,anchor+'\n'+flag,1)
        source=source.replace('    failed_host_ids: set[int] = set()','    failed_host_ids: set[int] = set(getattr(args, "exclude_host_id", None) or [])',1)
    old='''                if instance_id is not None:
                    try:
                        destroy_instance(instance_id)
                    except Exception as destroy_error:
                        print_stderr(f"warning: failed to destroy instance {instance_id}: {destroy_error}")'''
    new='''                if instance_id is not None:
                    from vast_failure_recovery import cleanup_infrastructure_failure
                    recovered_cleanup = cleanup_infrastructure_failure(
                        instance_id, connection, read_json(paths["metadata"]).get("remote_workdir", ""),
                        sync_paths, paths, config.ssh_private_key, attempt_record,
                        fetch_logs=fetch_vast_logs, sync_outputs=sync_remote_paths,
                        destroy=destroy_instance, update=update_metadata)
                    if not recovered_cleanup:
                        # Preserve instance_id so the owner's finally block retries
                        # cleanup rather than discarding the only live reference.
                        raise InfrastructureError("Infrastructure cleanup unconfirmed; owner finalizer will retry")'''
    if 'recovered_cleanup = cleanup_infrastructure_failure' not in source:
        if old not in source:raise ValueError('Unsupported infrastructure cleanup block')
        source=source.replace(old,new,1)
    signature='''    local_destination: Path,
) -> list[str]:'''
    if 'transfer_timeout_seconds: float | None = None' not in source:
        if signature not in source:raise ValueError('Unsupported artifact sync signature')
        source=source.replace(signature,'''    local_destination: Path,
    transfer_timeout_seconds: float | None = None,
) -> list[str]:''',1)
        start=source.index('def sync_remote_paths(');end=source.index('\ndef fetch_vast_logs(',start)
        section=source[start:end]
        section=section.replace('            check=False,\n        )','            check=False,\n            timeout=transfer_timeout_seconds,\n        )',1)
        old_body='''        assert archive_process.stdout is not None
        with tarfile.open(fileobj=archive_process.stdout, mode="r|gz") as tar:
            tar.extractall(local_destination)
        stderr = archive_process.stderr.read().decode() if archive_process.stderr else ""
        if archive_process.wait() != 0:
            raise InfrastructureError(f"Failed to sync remote path {sync_path}.\\nstderr:\\n{stderr}")
        synced.append(sync_path)'''
        new_body='''        assert archive_process.stdout is not None
        timer = None
        if transfer_timeout_seconds is not None:
            import threading
            timer = threading.Timer(transfer_timeout_seconds, archive_process.kill)
            timer.daemon = True
            timer.start()
        try:
            with tarfile.open(fileobj=archive_process.stdout, mode="r|gz") as tar:
                tar.extractall(local_destination)
            stderr = archive_process.stderr.read().decode() if archive_process.stderr else ""
            if archive_process.wait() != 0:
                raise InfrastructureError(f"Failed to sync remote path {sync_path}.\\nstderr:\\n{stderr}")
            synced.append(sync_path)
        finally:
            if timer is not None:
                timer.cancel()
            if archive_process.poll() is None:
                archive_process.kill()
                archive_process.wait()'''
        if old_body not in section:raise ValueError('Unsupported sync body')
        section=section.replace(old_body,new_body,1);source=source[:start]+section+source[end:]
    command_start=source.index('def run_command(');command_end=source.index('\ndef command_failure_message',command_start)
    command=source[command_start:command_end]
    if 'timeout: float | None = None' not in command:
        command=command.replace('    input_text: str | None = None,','    input_text: str | None = None,\n    timeout: float | None = None,',1)
        command=command.replace('        capture_output=capture_output,','        capture_output=capture_output,\n        timeout=timeout,',1)
        source=source[:command_start]+command+source[command_end:]
    # Bound provider log retrieval too, so it cannot delay failure cleanup forever.
    source=source.replace('result = run_command(["vastai", "logs", str(instance_id)], check=False)',
        'result = run_command(["vastai", "logs", str(instance_id)], check=False, timeout=45)')
    source=source.replace('run_command(["vastai", "destroy", "instance", str(instance_id), "-y"])',
        'run_command(["vastai", "destroy", "instance", str(instance_id), "-y"], timeout=45)')
    compile(source,'vast_run.py','exec')
    return source


def install(path):
    path=Path(path);before=path.read_text();after=patch(before)
    if before!=after:
        backup=path.with_suffix('.before-recovery.py')
        if not backup.exists():shutil.copy2(path,backup)
        path.write_text(after)
    shutil.copyfile(Path(__file__).with_name('failure_recovery.py'),path.with_name('vast_failure_recovery.py'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runner',required=True);a=p.parse_args();install(a.runner)
