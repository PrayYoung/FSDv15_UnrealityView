"""Require every frozen source, split and native-front cache before a full pilot."""
import argparse
import json
from pathlib import Path
from unrealityview.data.cache_recovery import recoverable_episodes


def validate(root):
    selection=json.loads(Path('configs/pandaset_validated_pilot.json').read_text())
    recovered=recoverable_episodes(root,selection['revision'],selection['sequences'],selection['frames'])
    expected={f"pandaset/{e['sequence']}" for e in selection['sequences']}
    if set(recovered)!=expected:raise ValueError('Full pilot requires all 76 verified frozen caches')
    print('FULL_FROZEN_CACHE_OK scenes=76')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);a=p.parse_args();validate(a.data)
