"""Bounded, checked artifact retrieval, called only by the global Vast owner."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time


class ArtifactError(RuntimeError):
    pass


def validate_return(root):
    files = [p for p in root.rglob('*') if p.is_file()]
    if not files or not any(p.stat().st_size for p in files):
        raise ArtifactError('Artifact return is empty')
    manifest = root / 'artifact_manifest.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        expected = record['files']
        if not expected:
            raise ArtifactError('Artifact manifest is empty')
        actual = {p.relative_to(root).as_posix() for p in files if p != manifest}
        if actual != set(expected):
            raise ArtifactError('Artifact manifest file set mismatch')
        for name, info in expected.items():
            p = root / name
            if p.stat().st_size != info['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != info['sha256']:
                raise ArtifactError('Artifact checksum mismatch: ' + name)


def extract_checked(stream, destination):
    # Accept bounded login banners, but never extract links or paths outside staging.
    from vast_lifecycle import skip_login_banner
    with tarfile.open(fileobj=skip_login_banner(stream), mode='r|gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ArtifactError('Unsafe remote archive member: ' + member.name)
            archive.extract(member, destination, filter='data')


def sync_paths(ssh_argv, remote_workdir, paths, destination, *, timeout_seconds=120,
               attempts=3, execute=subprocess.run, popen=subprocess.Popen, sleep=time.sleep):
    """Retry retrieval of an existing job only. Never rent, relaunch or extend a cap.

    No failed probe is interpreted as an absent optional output. Both failed and
    empty returns raise, so the owner cannot claim artifact success. Existing
    local artifacts are replaced only after a complete verified extraction.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    synced = []
    for relative in paths:
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts or str(path) == '.':
            raise ArtifactError('Invalid artifact path')
        target = remote_workdir.rstrip('/') + '/' + str(path)
        errors = []
        for attempt in range(attempts):
            process = None
            timer = None
            try:
                probe = execute([*ssh_argv, 'bash', '-s'], input='test -d ' + shlex.quote(target) + '\n',
                                text=True, capture_output=True, check=False, timeout=min(30, timeout_seconds))
                if probe.returncode:
                    raise ArtifactError(f'Artifact probe exit {probe.returncode}: {probe.stderr[-400:]}')
                with tempfile.TemporaryDirectory(prefix='.artifact-', dir=destination) as staging:
                    # A file for stderr avoids a full pipe blocking archive reads.
                    with tempfile.TemporaryFile() as stderr:
                        process = popen([*ssh_argv, 'tar', '-C', remote_workdir, '-czf', '-', str(path)],
                                        stdout=subprocess.PIPE, stderr=stderr)
                        timer = threading.Timer(timeout_seconds, process.kill)
                        timer.daemon = True
                        timer.start()
                        extract_checked(process.stdout, Path(staging))
                        code = process.wait(timeout=timeout_seconds)
                        if code:
                            stderr.seek(0)
                            raise ArtifactError(f'Artifact stream exit {code}: {stderr.read(400)!r}')
                    returned = Path(staging) / str(path)
                    validate_return(returned)
                    local = destination / str(path)
                    local.mkdir(parents=True, exist_ok=True)
                    # Keep a previous valid return if validation above fails.
                    for file in returned.rglob('*'):
                        if file.is_file():
                            output = local / file.relative_to(returned)
                            output.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(file, output)
                    synced.append(relative)
                    break
            except (OSError, EOFError, ValueError, KeyError, tarfile.TarError, subprocess.SubprocessError, ArtifactError) as error:
                errors.append(f'attempt {attempt+1}: {error}')
                if attempt + 1 < attempts:
                    sleep(2)
            finally:
                if timer:
                    timer.cancel()
                if process:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    if process.stdout:
                        process.stdout.close()
        else:
            raise ArtifactError(f'Could not retrieve {relative}; ' + '; '.join(errors))
    return synced
