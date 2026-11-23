"""Select camera/JSON members through HTTP Range; never download the full 44GB ZIP.

The public georghess/PandaSet mirror is unofficial; source/revision and ZIP CRCs
are recorded. Only JPG/JSON members under the six cameras are ever extracted.
No pickle, LiDAR or annotations are opened. Byte and time limits are explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import zipfile
import fsspec
from huggingface_hub import HfApi
from unrealityview.data.cache import atomic_json,resolve_safe_subpath
from unrealityview.data.real_world import PANDA_CAMERAS
from unrealityview.data.splits import stable_split,validate_split_manifest


def selected_sequences(names,seed=42,train_count=2,val_count=1,test_count=1,all_sequences=False):
    ids=sorted({PurePosixPath(name).parts[-4] for name in names if name.endswith("camera/front_camera/timestamps.json")})
    result=[]
    for split,count in (("train",train_count),("val",val_count),("test",test_count)):
        matching=[s for s in ids if stable_split("pandaset",s,seed)==split]
        if all_sequences:count=len(matching)
        if count<0:raise ValueError("Sequence counts must be nonnegative")
        if len(matching)<count:raise ValueError(f"Not enough sequences for {split}")
        result.extend((s,split) for s in matching[:count])
    if not result:raise ValueError("No sequences selected")
    return result


def fetch(output,manifest_path,max_frames=80,max_bytes=2*2**30,seed=42,train_count=2,val_count=1,test_count=1,all_sequences=False):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    revision=HfApi(token=False).dataset_info("georghess/pandaset").sha
    url=f"https://huggingface.co/datasets/georghess/pandaset/resolve/{revision}/pandaset.zip"
    extracted=0;records=[]
    with fsspec.open(url,"rb",block_size=1024*1024).open() as remote:
        with zipfile.ZipFile(remote) as archive:
            selections=selected_sequences(archive.namelist(),seed,train_count,val_count,test_count,all_sequences)
            for sequence,split in selections:
                prefix=f"pandaset/{sequence}/camera/"
                members=[]
                for info in archive.infolist():
                    if not info.filename.startswith(prefix) or info.is_dir():continue
                    parts=PurePosixPath(info.filename).parts
                    if len(parts)!=5 or parts[3] not in PANDA_CAMERAS:continue
                    filename=parts[4]
                    if filename not in ("intrinsics.json","poses.json","timestamps.json"):
                        if not filename.endswith(".jpg") or not filename[:-4].isdigit() or int(filename[:-4])>=max_frames:continue
                    members.append(info)
                if extracted+sum(m.file_size for m in members)>max_bytes:raise ValueError("Camera subset exceeds extraction byte budget")
                for info in members:
                    relative=PurePosixPath(info.filename).relative_to("pandaset")
                    target=resolve_safe_subpath(root,str(relative));target.parent.mkdir(parents=True,exist_ok=True)
                    if info.file_size>32*2**20:raise ValueError("Unexpectedly large image/JSON ZIP member")
                    with archive.open(info) as source,target.open("wb") as destination:shutil.copyfileobj(source,destination)
                    if target.suffix==".json" and target.name!="intrinsics.json":
                        values=json.loads(target.read_text())[:max_frames]
                        atomic_json(target,values)
                    extracted+=info.file_size
                    records.append({"member":info.filename,"crc32":info.CRC,"bytes":info.file_size})
                print(f"FETCHED pandaset/{sequence} split={split} members={len(members)} extracted_mib={extracted/2**20:.1f}",flush=True)
    manifest=validate_split_manifest({"version":1,"seed":seed,"source":"PandaSet camera-only",
                    "download":{"mirror":"georghess/pandaset","revision":revision,"extracted_bytes":extracted,
                                "transport":"HTTP ranges; selected camera JPG/JSON only","members":records},
                    "sequences":[dict(dataset="pandaset",sequence=s,split=split,root=str(root)) for s,split in selections]})
    atomic_json(manifest_path,manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default=".cache/pandaset_native")
    p.add_argument("--manifest",default="outputs/pandaset_smoke/splits.json")
    p.add_argument("--frames",type=int,default=80)
    p.add_argument("--train-count",type=int,default=2);p.add_argument("--val-count",type=int,default=1)
    p.add_argument("--test-count",type=int,default=1);p.add_argument("--all",action="store_true",dest="all_sequences")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-gib",type=float,default=2.)
    a=p.parse_args()
    if not 48<=a.frames<=80:p.error("frames must be 48–80")
    if not 0<a.max_gib<=25:p.error("max-gib must be >0 and <=25")
    if min(a.train_count,a.val_count,a.test_count)<0:p.error("Sequence counts must be nonnegative")
    fetch(a.output,a.manifest,a.frames,int(a.max_gib*2**30),a.seed,a.train_count,a.val_count,a.test_count,a.all_sequences)


if __name__=="__main__":main()
