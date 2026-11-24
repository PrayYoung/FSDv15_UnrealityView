"""Fail closed before training if the measured estimate exceeds the frozen time cap."""
import argparse
import json
from pathlib import Path
from unrealityview.data.cache import atomic_json


def decide(reports):
    passed=[x for x in reports if x['status']=='passed']
    if len(passed)!=1:raise ValueError('Need exactly one successful preflight')
    chosen=passed[0]
    # 40% allowance for periodic evaluations, checkpoint writing and variability.
    estimate=chosen['projected_training_seconds']*1.4
    record=dict(chosen=chosen,conservative_training_seconds=estimate,training_timeout_seconds=14400,
        proceed=estimate<=14400,automatic_grid_fallback=False)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory)
    reports=[json.loads(x.read_text()) for x in sorted(root.glob('preflight_batch*.json'))]
    record=decide(reports)
    atomic_json(root/'preflight_decision.json',record)
    if not record['proceed']:raise RuntimeError('Measured projection exceeds cap; report before any 5x8 fallback')


if __name__=='__main__':main()
