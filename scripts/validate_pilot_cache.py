"""Require every frozen source, split and native-front cache before a full pilot."""
import argparse
import json
from pathlib import Path
from unrealityview.data.cache_recovery import recoverable_episodes


def validate(root,training_only=False,training_inputs_only=False):
    selection=json.loads(Path('configs/pandaset_validated_pilot.json').read_text())
    entries=[e for e in selection['sequences'] if not training_only or e['split']!='test']
    recovered=recoverable_episodes(root,selection['revision'],entries,selection['frames'],**({'raw_cameras':('chase',)} if training_inputs_only else {}))
    expected={f"pandaset/{e['sequence']}" for e in entries}
    if set(recovered)!=expected:raise ValueError('Pilot requires every selected frozen cache')
    print(f'FULL_FROZEN_CACHE_OK scenes={len(expected)} training_only={training_only}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--training-only',action='store_true');p.add_argument('--training-inputs-only',action='store_true');a=p.parse_args();validate(a.data,a.training_only,a.training_inputs_only)
