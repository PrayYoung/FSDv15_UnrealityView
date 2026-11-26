"""Small lifecycle helpers used only by the global Vast owner."""
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone


def prevent_idle_sleep():
    if sys.platform!='darwin':return None
    executable=shutil.which('caffeinate')
    if not executable:raise RuntimeError('caffeinate required before cost-capped macOS rentals')
    return subprocess.Popen([executable,'-i','-s','-w',str(os.getpid())],
                            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def wait_wall_deadline(deadline,now=lambda:datetime.now(timezone.utc),sleep=time.sleep):
    # Recheck wall time after wake; never sleep the full original duration again.
    while (remaining:=(deadline-now()).total_seconds())>0:sleep(min(5.,remaining))


def skip_login_banner(stream,max_prefix=65536):
    """Strip bounded shell banner bytes before the gzip magic; never parse text."""
    prefix=bytearray()
    while len(prefix)<max_prefix:
        value=stream.read(1)
        if not value:raise ValueError('Remote archive ended before gzip header')
        prefix.extend(value)
        if prefix[-3:]==b'\x1f\x8b\x08':return PrefixStream(bytes(prefix[-3:]),stream)
    raise ValueError('Remote archive banner exceeded limit')


class PrefixStream:
    def __init__(self,prefix,stream):self.prefix=prefix;self.stream=stream
    def read(self,size=-1):
        if size<0:result=self.prefix+self.stream.read();self.prefix=b'';return result
        first=self.prefix[:size];self.prefix=self.prefix[len(first):]
        return first+self.stream.read(size-len(first))
