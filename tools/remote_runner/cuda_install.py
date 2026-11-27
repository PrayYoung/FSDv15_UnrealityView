"""Add an opt-in CUDA capability filter before the Vast owner rents a host."""
import argparse
from pathlib import Path
import shutil


def patch(source):
    if '--min-cuda-version' in source:
        return source
    source=source.replace('    excluded_host_ids: set[int] | None = None,',
        '    excluded_host_ids: set[int] | None = None,\n    min_cuda_version: float | None = None,',1)
    anchor='    offers = vast_json(\n'
    source=source.replace(anchor,'    if min_cuda_version is not None:\n        query += f" cuda_max_good>={min_cuda_version}"\n'+anchor,1)
    source=source.replace('        and int(offer.get("host_id", -1)) not in excluded_host_ids',
        '        and int(offer.get("host_id", -1)) not in excluded_host_ids\n        and (min_cuda_version is None or float(offer.get("cuda_max_good") or 0) >= min_cuda_version)',1)
    for anchor in ('                excluded_host_ids=failed_host_ids,','                    excluded_host_ids=set(),'):
        if anchor not in source:raise ValueError('CUDA filter call anchor changed')
        indent=anchor[:len(anchor)-len(anchor.lstrip())]
        source=source.replace(anchor,anchor+'\n'+indent+'min_cuda_version=getattr(args, "min_cuda_version", None),',1)
    anchor='    run_parser.add_argument("--dry-run-search", action="store_true")'
    if anchor not in source:raise ValueError('CUDA filter parser anchor changed')
    source=source.replace(anchor,anchor+'\n    run_parser.add_argument("--min-cuda-version", type=float, help="Require advertised host CUDA support before renting")',1)
    compile(source,'vast_run.py','exec')
    return source


def install(path):
    path=Path(path);before=path.read_text();after=patch(before)
    if before!=after:
        backup=path.with_suffix('.before-cuda-filter.py')
        if not backup.exists():shutil.copy2(path,backup)
        path.write_text(after)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runner',required=True)
    install(p.parse_args().runner)
