"""Fetch and verify only the pinned upstream pipeline source (not model weights)."""
import hashlib,json
from pathlib import Path
from urllib.request import urlopen


def main():
    spec=json.loads(Path('configs/difix_runtime_pins.json').read_text())
    with urlopen(spec['pipeline_url'],timeout=30) as response:payload=response.read(2**20)
    if hashlib.sha256(payload).hexdigest()!=spec['pipeline_sha256']:raise ValueError('Difix pipeline fingerprint mismatch')
    path=Path('.cache/third_party/difix/src/pipeline_difix.py');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    print('DIFIX_PINNED_SOURCE_VERIFIED')

if __name__=='__main__':main()
