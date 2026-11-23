"""Restore verified completed episodes into a newly frozen valid-scene manifest."""
import argparse
import json
from pathlib import Path
from unrealityview.data.cache_recovery import seed_recovery

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--recovery',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
a=p.parse_args();count=seed_recovery(a.recovery,a.output,json.loads(Path(a.manifest).read_text()))
print(f'RESTORED_ENCODED_EPISODES {count}',flush=True)
