"""Bound native build parallelism by host/container RAM and require Python headers."""
import argparse
import os
from pathlib import Path
import sysconfig


def build_jobs(memory_bytes):
    return 2 if memory_bytes<24*2**30 else 4


def memory_limit():
    total=os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    for path in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            text=Path(path).read_text().strip()
            if text.isdigit() and int(text)>0:total=min(total,int(text))
        except OSError:pass
    return total


def check_headers():
    header=Path(sysconfig.get_path('include'))/'Python.h'
    if not header.is_file():raise RuntimeError('Python development headers missing for the selected interpreter; install matching python3.10-dev before package builds')
    return header


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',action='store_true');p.add_argument('--check-headers',action='store_true');a=p.parse_args()
    if a.jobs:print(build_jobs(memory_limit()))
    if a.check_headers:check_headers();print('PYTHON_BUILD_HEADERS_OK')
