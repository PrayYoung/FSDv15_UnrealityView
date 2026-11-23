"""Download one immutable archive using an isolated modern Hub/Xet client.

Incomplete files remain in local_dir for SDK resume. No archive is copied to the
training output directory. Only public source identity and checksums are logged.
"""
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def verify_archive(path, spec):
    if path.stat().st_size != spec['bytes']:
        raise ValueError('Archive size mismatch')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(chunk)
    if digest.hexdigest() != spec['sha256']:
        raise ValueError('Archive SHA256 mismatch')
    return digest.hexdigest()


def retry_delay(error, attempt):
    response = getattr(error, 'response', None)
    status = getattr(response, 'status_code', None)
    if status is not None and status not in (408, 429, 500, 502, 503, 504):
        return None
    if status is None:
        import httpx
        if not isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError)):
            return None
    headers = getattr(response, 'headers', {})
    delay = min(30 * 2**attempt, 300)
    retry = headers.get('Retry-After')
    if retry:
        try:
            delay = max(delay, float(retry))
        except ValueError:
            delay = max(delay, (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
    # Refuse an excessive wait rather than hammering the server before its reset.
    return delay if delay <= 600 else None


def download(spec, directory, record, downloader=None, sleep=time.sleep):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec['filename']
    start = time.monotonic()
    if not target.exists():
        # Reserve space for runtime, selected cameras, model and encoded caches.
        # Count physically allocated partial blocks, not a sparse logical size.
        partial = min(spec['bytes'], sum(p.stat().st_blocks * 512
            for p in directory.rglob('*.incomplete') if p.is_file()))
        if shutil.disk_usage(directory).free < spec['bytes'] - partial + 25 * 2**30:
            raise OSError('Insufficient free space: archive plus 25 GiB reserve required')
        if downloader is None:
            from huggingface_hub import hf_hub_download
            downloader = hf_hub_download
        for attempt in range(4):
            try:
                downloaded = downloader(repo_id=spec['repo'], filename=spec['filename'],
                    repo_type='dataset', revision=spec['revision'], local_dir=str(directory), token=False)
                if Path(downloaded).resolve() != target.resolve():
                    raise ValueError('Unexpected SDK download location')
                break
            except Exception as error:
                delay = retry_delay(error, attempt)
                if attempt == 3 or delay is None:
                    # SDK exceptions can include signed URLs; emit only safe classification.
                    raise RuntimeError('Archive download failed: ' + type(error).__name__) from None
                print(f'ARCHIVE_RETRY attempt={attempt+1} wait_seconds={delay:.0f}', flush=True)
                sleep(delay)
    checksum = verify_archive(target, spec)
    result = dict(spec, verified_sha256=checksum, elapsed_seconds=time.monotonic()-start)
    record = Path(record); record.parent.mkdir(parents=True, exist_ok=True)
    temporary = record.with_suffix('.part')
    temporary.write_text(json.dumps(result, indent=2)+'\n'); os.replace(temporary, record)
    print(f'ARCHIVE_VERIFIED bytes={spec["bytes"]} seconds={result["elapsed_seconds"]:.1f}', flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', default='configs/pandaset_archive.json')
    parser.add_argument('--directory', default='.cache/pandaset_archive')
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    download(json.loads(Path(args.spec).read_text()), args.directory, args.record)


if __name__ == '__main__':
    main()
