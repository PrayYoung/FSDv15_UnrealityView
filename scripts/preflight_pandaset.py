"""CPU-only metadata validation and deterministic valid-scene selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path,PurePosixPath
import zipfile
import fsspec
from huggingface_hub import HfApi
from unrealityview.data.cache import atomic_json
from unrealityview.data.real_world import PANDA_CAMERAS
from unrealityview.data.pandaset_preflight import validate_camera_metadata
from unrealityview.data.splits import stable_split


def inspect_sequence(url,sequence,frames,cache):
    path=cache/(sequence+'.json')
    if path.is_file():return json.loads(path.read_text())
    cameras={}
    try:
        with fsspec.open(url,'rb',block_size=65536).open() as remote:
            with zipfile.ZipFile(remote) as archive:
                names=archive.namelist()
                for camera in PANDA_CAMERAS:
                    prefix=f'pandaset/{sequence}/camera/{camera}/'
                    read=lambda name:json.loads(archive.read(prefix+name+'.json'))
                    indices=sorted(int(PurePosixPath(n).stem) for n in names
                        if n.startswith(prefix) and len(PurePosixPath(n).parts)==5 and n.endswith('.jpg')
                        and PurePosixPath(n).stem.isdigit() and int(PurePosixPath(n).stem)<frames)
                    cameras[camera]=dict(intrinsics=read('intrinsics'),poses=read('poses'),
                                         timestamps=read('timestamps'),image_indices=indices)
        metrics=validate_camera_metadata(cameras,frames)
        result=dict(sequence=sequence,valid=True,**metrics)
    except (ValueError,KeyError,TypeError) as error:
        result=dict(sequence=sequence,valid=False,error=str(error))
    atomic_json(path,result);return result


def preflight(output,frames=80,counts=(60,8,8),seed=42,workers=8,revision=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    revision=revision or HfApi(token=False).dataset_info('georghess/pandaset').sha
    url=f'https://huggingface.co/datasets/georghess/pandaset/resolve/{revision}/pandaset.zip'
    cache=output/'metadata_checks'/revision/str(frames);cache.mkdir(parents=True,exist_ok=True)
    with fsspec.open(url,'rb',block_size=65536).open() as remote:
        with zipfile.ZipFile(remote) as archive:
            ids=sorted({PurePosixPath(n).parts[-4] for n in archive.namelist() if n.endswith('camera/front_camera/timestamps.json')})
    rows={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(inspect_sequence,url,sequence,frames,cache):sequence for sequence in ids}
        for future in as_completed(futures):rows[futures[future]]=future.result()
    selected=[];available={}
    for split,count in zip(('train','val','test'),counts):
        candidates=[s for s in ids if rows[s]['valid'] and stable_split('pandaset',s,seed)==split]
        available[split]=len(candidates)
        selected.extend(dict(dataset='pandaset',sequence=s,split=split) for s in candidates[:count])
    report=dict(revision=revision,frames=frames,seed=seed,available_valid=available,
                rejected=[rows[s] for s in ids if not rows[s]['valid']],checked=len(ids),sequences=selected,
                policy='Reject invalid whole scenes; preserve clocks, calibration and stable hash partitions')
    atomic_json(output/'preflight.json',report)
    if any(available[split]<count for split,count in zip(('train','val','test'),counts)):
        raise ValueError(f'Insufficient valid scenes: {available}; review preflight.json before changing counts')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--frames',type=int,default=80);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--revision');a=p.parse_args()
    report=preflight(a.output,a.frames,workers=a.workers,revision=a.revision)
    print(json.dumps({k:v for k,v in report.items() if k!='sequences'},indent=2))


if __name__=='__main__':main()
