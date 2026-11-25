"""Record a bounded scene-verification stage without exposing command arguments."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from unrealityview.data.cache import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--stage',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:p.error('Command required')
    start=time.monotonic();code=None
    try:code=subprocess.call(command)
    finally:
        path=Path(a.record);data=json.loads(path.read_text()) if path.exists() else {}
        data[a.stage]=dict(seconds=time.monotonic()-start,exit_code=code);atomic_json(path,data)
    raise SystemExit(code)


if __name__=='__main__':main()
