"""Bind the completed free local parity probe to one training episode; no DINO run."""
import argparse
import json
from pathlib import Path
from scripts.reencode_uve_cache import digest
from unrealityview.data.cache import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--report',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    report=json.loads(Path(a.report).read_text());matches=[]
    for meta in (Path(a.data)/'train').glob('*/metadata.json'):
        if json.loads(meta.read_text()).get('source_id')==report['source_id']:matches.append(meta.parent/'episode.safetensors')
    if len(matches)!=1 or digest(matches[0])!=report['source_episode_sha256']:raise ValueError('Local parity reference no longer matches training episode')
    atomic_json(a.output,dict(source_id=report['source_id'],source_episode_sha256=report['source_episode_sha256'],local_probe_paid_cost_usd=0,
        historical_parity_passed=False,note='Completed local diagnosis; mismatch remains unresolved. Arm E uses original4x8 features unchanged.'))


if __name__=='__main__':main()
