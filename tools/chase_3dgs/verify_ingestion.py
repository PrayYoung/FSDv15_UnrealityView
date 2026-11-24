"""Exercise the core teacher cache importer on one real scene after export."""
import argparse
from pathlib import Path
import json
from prepare_real_data import target_frame,cache_manifest
from unrealityview.data.splits import stable_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',required=True);p.add_argument('--targets',required=True)
    p.add_argument('--sequence',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    entry=dict(dataset='pandaset',sequence=a.sequence,root=str(Path(a.native).resolve()),split=stable_split('pandaset',a.sequence))
    times=json.loads((Path(a.native)/a.sequence/'camera/front_camera/timestamps.json').read_text())
    for i,t in enumerate(times):target_frame(a.targets,entry,i,t)
    cache_manifest(dict(version=1,seed=42,sequences=[entry]),a.output,device='cuda',target='teacher',targets=a.targets)


if __name__=='__main__':main()
