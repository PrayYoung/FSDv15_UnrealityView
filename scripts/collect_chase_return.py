"""Whitelist small scene-verification artifacts; never return native data or weights."""
import argparse
import hashlib
from pathlib import Path
import shutil
from unrealityview.data.cache import atomic_json


def collect(source,destination,sequence,exit_code=None):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination or source in destination.parents or destination in source.parents:raise ValueError('Separate working and return directories required')
    if len(sequence)!=3 or not sequence.isdigit():raise ValueError('Invalid sequence')
    files=['archive_record.json','stage_times.json']
    scene=Path('pandaset')/sequence
    files += [str(scene/name) for name in ('manifest.json','quality.json','calibration.json','timings.json','inspection.png','chase_raw.gif','chase_difix.gif')]
    files += [str(scene/'rgb'/f'{i:05d}.png') for i in range(80)]
    ingestion=source/'ingestion_check'
    if ingestion.exists():files += [str(p.relative_to(source)) for p in ingestion.rglob('*.json')]
    records=[]
    for relative in files:
        path=source/relative
        if not path.exists():continue
        if path.is_symlink() or not path.is_file() or source not in path.resolve().parents:raise ValueError('Invalid artifact path')
        if path.stat().st_size>8*2**20:raise ValueError('Unexpectedly large diagnostic artifact')
        target=destination/relative
        if target.is_symlink() or destination not in target.resolve().parents:raise ValueError('Invalid return path')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        records.append(dict(path=relative,bytes=path.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    atomic_json(destination/'collection.json',dict(sequence=sequence,workload_exit_code=exit_code,files=records,
        total_bytes=sum(r['bytes'] for r in records),excluded=['native_archive','native_data','alpha','build_environments','fit_checkpoints','ingestion_tensors_and_RGB']))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True);p.add_argument('--sequence',required=True);p.add_argument('--exit-code',type=int);a=p.parse_args()
    collect(a.source,a.destination,a.sequence,a.exit_code)


if __name__=='__main__':main()
