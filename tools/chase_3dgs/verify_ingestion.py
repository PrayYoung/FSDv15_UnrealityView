"""Exercise the core teacher cache importer on one real scene after export."""
import argparse
from pathlib import Path
import json
import time
from unrealityview.data.cache import atomic_json
from prepare_real_data import target_frame,cache_manifest
from unrealityview.data.splits import stable_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',required=True);p.add_argument('--targets',required=True)
    p.add_argument('--sequence',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    entry=dict(dataset='pandaset',sequence=a.sequence,root=str(Path(a.native).resolve()),split=stable_split('pandaset',a.sequence))
    times=json.loads((Path(a.native)/a.sequence/'camera/front_camera/timestamps.json').read_text())
    for i,t in enumerate(times):target_frame(a.targets,entry,i,t)
    start=time.monotonic()
    cache_manifest(dict(version=1,seed=42,sequences=[entry]),a.output,device='cuda',target='teacher',targets=a.targets)
    metadata=[json.loads(p.read_text()) for p in Path(a.output).glob('*/*/metadata.json')]
    if len(metadata)!=1 or metadata[0]['target_view']!='elevated_chase' or metadata[0]['frames']!=len(times):raise ValueError('Ingestion output identity/shape mismatch')
    atomic_json(Path(a.output)/'ingestion_summary.json',dict(status='passed',source_id=f'pandaset/{a.sequence}',target_view='elevated_chase',frames=len(times),seconds=time.monotonic()-start))


if __name__=='__main__':main()
