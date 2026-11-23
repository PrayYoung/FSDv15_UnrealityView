"""Select camera/JSON members through HTTP Range; never download the full 44GB ZIP.

The public georghess/PandaSet mirror is unofficial; source/revision and ZIP CRCs
are recorded. Only JPG/JSON members under the six cameras are ever extracted.
No pickle, LiDAR or annotations are opened. Byte and time limits are explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import os
from concurrent.futures import ThreadPoolExecutor,as_completed
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


def camera_members(archive,sequence,max_frames):
    prefix=f"pandaset/{sequence}/camera/";members=[]
    for info in archive.infolist():
        if not info.filename.startswith(prefix) or info.is_dir():continue
        parts=PurePosixPath(info.filename).parts
        if len(parts)!=5 or parts[3] not in PANDA_CAMERAS:continue
        filename=parts[4]
        if filename not in ("intrinsics.json","poses.json","timestamps.json"):
            if not filename.endswith(".jpg") or not filename[:-4].isdigit() or int(filename[:-4])>=max_frames:continue
        if info.file_size>32*2**20:raise ValueError("Unexpectedly large image/JSON ZIP member")
        members.append(info)
    if len({m.filename for m in members})!=len(members):raise ValueError("Duplicate camera ZIP members")
    return members


def download_sequence(url,root,sequence,max_frames,expected,resume=True):
    """Each worker owns its HTTP cursor and ZipFile; never share seek state."""
    ledger_path=root/".fetch_state"/(sequence+".json")
    previous=json.loads(ledger_path.read_text()) if resume and ledger_path.is_file() else {}
    completed={};records=[];reused=0;fetched=0
    with fsspec.open(url,"rb",block_size=1024*1024).open() as remote:
        with zipfile.ZipFile(remote) as archive:
            members=camera_members(archive,sequence,max_frames)
            identity=[(m.filename,m.CRC,m.file_size) for m in members]
            if identity!=expected:raise ValueError("ZIP inventory changed during download")
            for index,info in enumerate(members):
                relative=PurePosixPath(info.filename).relative_to("pandaset")
                target=resolve_safe_subpath(root,str(relative));target.parent.mkdir(parents=True,exist_ok=True)
                prior=previous.get(info.filename)
                valid=(prior and prior["crc32"]==info.CRC and prior["source_bytes"]==info.file_size
                       and target.is_file() and target.stat().st_size==prior["saved_bytes"]
                       and hashlib.sha256(target.read_bytes()).hexdigest()==prior["sha256"])
                if valid:
                    completed[info.filename]=prior;reused+=1
                else:
                    # read() verifies ZIP CRC before any completed file is published.
                    payload=archive.read(info)
                    if target.suffix==".json" and target.name!="intrinsics.json":
                        values=json.loads(payload)
                        if not isinstance(values,list):raise ValueError("Expected a native JSON frame array")
                        payload=(json.dumps(values[:max_frames],indent=2,allow_nan=False)+"\n").encode()
                    temporary=target.with_name(target.name+".part")
                    temporary.write_bytes(payload);os.replace(temporary,target);fetched+=1
                    completed[info.filename]=dict(crc32=info.CRC,source_bytes=info.file_size,
                         saved_bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
                records.append(dict(member=info.filename,crc32=info.CRC,bytes=info.file_size))
                if index%32==31:atomic_json(ledger_path,completed)
    atomic_json(ledger_path,completed)
    return records,reused,fetched


def fetch(output,manifest_path,max_frames=80,max_bytes=2*2**30,seed=42,train_count=2,val_count=1,test_count=1,
          all_sequences=False,workers=4,resume=True):
    if not 1<=workers<=8:raise ValueError("workers must be 1–8")
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    session_path=root/".fetch_state"/"session.json"
    prior=json.loads(session_path.read_text()) if session_path.is_file() else None
    revision=prior["revision"] if prior else HfApi(token=False).dataset_info("georghess/pandaset").sha
    url=f"https://huggingface.co/datasets/georghess/pandaset/resolve/{revision}/pandaset.zip"
    with fsspec.open(url,"rb",block_size=1024*1024).open() as remote:
        with zipfile.ZipFile(remote) as archive:
            selections=selected_sequences(archive.namelist(),seed,train_count,val_count,test_count,all_sequences)
            plans={sequence:[(m.filename,m.CRC,m.file_size) for m in camera_members(archive,sequence,max_frames)]
                   for sequence,_ in selections}
    total=sum(size for members in plans.values() for _,_,size in members)
    if total>max_bytes:raise ValueError("Camera subset exceeds extraction byte budget")
    session=dict(revision=revision,max_frames=max_frames,seed=seed,sequences=[list(p) for p in selections])
    if prior is not None and prior!=session:raise ValueError("Resume plan changed; use a new output directory")
    atomic_json(session_path,session)
    results={};extracted=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(download_sequence,url,root,sequence,max_frames,plans[sequence],resume):(sequence,split)
                 for sequence,split in selections}
        try:
            for future in as_completed(pending):
                sequence,split=pending[future];records,reused,fetched=future.result();results[sequence]=records
                extracted+=sum(r["bytes"] for r in records)
                print(f"FETCHED pandaset/{sequence} split={split} members={len(records)} reused={reused} "
                      f"downloaded={fetched} extracted_mib={extracted/2**20:.1f}",flush=True)
        except BaseException:
            for future in pending:future.cancel()
            raise
    # Completion order cannot change provenance or scene assignments.
    records=[r for sequence,_ in selections for r in results[sequence]]
    manifest=validate_split_manifest({"version":1,"seed":seed,"source":"PandaSet camera-only",
                    "download":{"mirror":"georghess/pandaset","revision":revision,"extracted_bytes":total,
                                "transport":"parallel HTTP ranges; selected camera JPG/JSON only","members":records},
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
    p.add_argument("--workers",type=int,default=4,help="Independent sequence downloads, 1–8")
    p.add_argument("--no-resume",action="store_true",help="Redownload rather than reuse hash-verified files")
    p.add_argument("--max-gib",type=float,default=2.)
    a=p.parse_args()
    if not 48<=a.frames<=80:p.error("frames must be 48–80")
    if not 0<a.max_gib<=25:p.error("max-gib must be >0 and <=25")
    if min(a.train_count,a.val_count,a.test_count)<0:p.error("Sequence counts must be nonnegative")
    if not 1<=a.workers<=8:p.error("workers must be 1–8")
    fetch(a.output,a.manifest,a.frames,int(a.max_gib*2**30),a.seed,a.train_count,a.val_count,a.test_count,a.all_sequences,a.workers,not a.no_resume)


if __name__=="__main__":main()
