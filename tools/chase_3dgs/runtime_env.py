"""Expose console tools installed beside the explicitly selected interpreter."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def configure_runtime_tools(executable=None,environment=None,check=True):
    environment=os.environ if environment is None else environment
    # Do not resolve the interpreter symlink: the venv's bin directory is needed.
    directory=Path(executable or sys.executable).absolute().parent
    environment['PATH']=str(directory)+os.pathsep+environment.get('PATH','')
    ninja=shutil.which('ninja',path=environment['PATH'])
    if ninja is None:raise RuntimeError('Ninja executable missing beside the selected Python environment')
    if check:subprocess.run([ninja,'--version'],check=True,timeout=10,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=environment)
    return ninja


if __name__=='__main__':
    p=argparse.ArgumentParser();p.parse_args();configure_runtime_tools();print('NINJA_EXECUTABLE_OK',flush=True)
