"""Seal the small return folder; print metrics into the durable worker log too."""
import hashlib
import json
from pathlib import Path
import sys


def seal(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Artifact symlinks are not supported')
        if path.is_file() and path.name != 'artifact_manifest.json':
            files[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if not files:
        raise ValueError('Empty return folder')
    manifest = dict(files=files)
    (root / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Preserve small summaries in the separately retrieved log if transfer fails.
    for name in ('summary.json', 'loss_diagnostics.json', 'evaluation.json', 'metrics.json', 'timings.json'):
        path = root / name
        if path.exists() and path.stat().st_size <= 32768:
            print('ARTIFACT_SUMMARY ' + name + ' ' + json.dumps(json.loads(path.read_text())), flush=True)
    print('ARTIFACT_SEALED ' + json.dumps(dict(files=len(files), bytes=sum(x['bytes'] for x in files.values()))), flush=True)
    return manifest


if __name__ == '__main__':
    seal(sys.argv[1])
