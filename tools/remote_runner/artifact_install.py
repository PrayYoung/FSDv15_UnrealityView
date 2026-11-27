"""Install verified artifact retrieval into the global owner; no cloud calls."""
import argparse
from pathlib import Path
import shutil


def patch(source):
    start = source.index('def sync_remote_paths(')
    end = source.index('\ndef fetch_vast_logs(', start)
    source = source[:start] + '''def sync_remote_paths(connection, ssh_private_key_path, remote_workdir, sync_paths,
                      local_destination, transfer_timeout_seconds=None):
    from vast_artifacts import sync_paths as retrieve
    return retrieve(ssh_base_command(connection, ssh_private_key_path), remote_workdir,
                    sync_paths, local_destination,
                    timeout_seconds=transfer_timeout_seconds or 120)

''' + source[end:]
    # Exercise the actual SSH archive path before spending time on training.
    if 'def artifact_transfer_preflight(' not in source:
        anchor = 'def sync_remote_paths('
        helper = '''def artifact_transfer_preflight(connection, key, workdir, local_run_dir):
    import uuid
    token = uuid.uuid4().hex
    relative = ".vast-transfer-check/" + token
    target = workdir.rstrip("/") + "/" + relative
    script = "mkdir -p " + shlex.quote(target) + "\\nprintf '%s' " + shlex.quote(token) + " > " + shlex.quote(target + "/receipt.txt") + "\\n"
    result = subprocess.run([*ssh_base_command(connection, key), "bash", "-s"],
                            input=script, text=True, capture_output=True, timeout=30, check=False)
    if result.returncode != 0:
        raise InfrastructureError("Artifact transfer preflight could not write receipt")
    local = Path(local_run_dir) / "transfer_preflight"
    sync_remote_paths(connection, key, workdir, (relative,), local, transfer_timeout_seconds=30)
    receipt = local / relative / "receipt.txt"
    if receipt.read_text() != token:
        raise InfrastructureError("Artifact transfer preflight receipt mismatch")


'''
        source = source.replace(anchor, helper + anchor, 1)
        anchor = '                remote_exit_code = ssh_run_workload('
        if anchor not in source:
            raise ValueError('Owner workload anchor changed')
        source = source.replace(anchor, '''                artifact_transfer_preflight(connection, config.ssh_private_key, remote_workdir, paths["run_dir"])
''' + anchor, 1)
    anchor = '                    print_stderr(f"warning: failed to sync remote outputs from instance {instance_id}: {error}")'
    if 'artifact_sync_error=str(error)' not in source:
        if anchor not in source:
            raise ValueError('Owner finalizer anchor changed')
        source = source.replace(anchor, anchor + '''
                    update_metadata(paths["metadata"], artifact_sync_error=str(error),
                                    final_classification="ARTIFACT_FAILED")
                    result_code = DEFAULT_INFRA_FAILURE_EXIT_CODE''', 1)
    compile(source, 'vast_run.py', 'exec')
    return source


def install(path):
    path = Path(path)
    before = path.read_text()
    after = patch(before)
    if after != before:
        backup = path.with_suffix('.before-artifacts.py')
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(after)
    shutil.copy2(Path(__file__).with_name('artifact_sync.py'), path.with_name('vast_artifacts.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', required=True)
    install(parser.parse_args().runner)
