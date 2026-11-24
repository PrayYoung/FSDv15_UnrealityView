"""Extract verified camera/LiDAR/cuboid bytes for isolated external preprocessing.

This tool never unpickles data. Only the external pinned PandaSet devkit reads
its native tables; none of those dependencies or formats enter the main package.
"""
import argparse
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import zipfile
from scripts.download_pandaset_archive import verify_archive

CAMERAS=('front_camera','front_left_camera','front_right_camera','left_camera','right_camera','back_camera')


def selected_members(archive,sequence):
    if len(sequence)!=3 or not sequence.isdigit():raise ValueError('PandaSet scene ID must contain three digits')
    prefix=f'pandaset/{sequence}/';members=[];names=set()
    for info in archive.infolist():
        if not info.filename.startswith(prefix) or info.is_dir():continue
        relative=PurePosixPath(info.filename).relative_to('pandaset');parts=relative.parts
        if '..' in parts or relative.is_absolute():raise ValueError('Unsafe ZIP path')
        sub=parts[1:]
        allowed=(len(sub)==3 and sub[0]=='camera' and sub[1] in CAMERAS and (sub[2].endswith('.jpg') or sub[2] in ('intrinsics.json','poses.json','timestamps.json')))
        allowed|=(len(sub)==2 and sub[0] in ('lidar','meta') and (sub[1].endswith('.json') or (sub[0]=='lidar' and sub[1].endswith('.pkl.gz'))))
        allowed|=(len(sub)==3 and sub[:2]==('annotations','cuboids') and sub[2].endswith(('.pkl.gz','.json')))
        if not allowed:continue
        if info.filename in names:raise ValueError('Duplicate ZIP member')
        if info.file_size>256*2**20:raise ValueError('Oversized native ZIP member')
        names.add(info.filename);members.append((info,relative))
    if not members:raise ValueError('Scene not present')
    return members


def extract(archive_path,output,sequences,spec,max_bytes=12*2**30,verified=False):
    if not verified:verify_archive(Path(archive_path),spec)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        plans=[(sequence,selected_members(archive,sequence)) for sequence in sequences]
        total=sum(info.file_size for _,members in plans for info,_ in members)
        if total>max_bytes:raise ValueError('Selected native scenes exceed extraction budget')
        if shutil.disk_usage(output).free<total+2**30:raise OSError('Insufficient native scratch space')
        for sequence,members in plans:
            for info,relative in members:
                target=(output/str(relative)).resolve()
                if output not in target.parents:raise ValueError('ZIP path escapes extraction root')
                target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.part')
                try:
                    with archive.open(info) as source,temporary.open('wb') as dest:shutil.copyfileobj(source,dest,2**20)
                    os.replace(temporary,target)
                finally:
                    if temporary.exists():temporary.unlink()
    record=dict(archive_sha256=spec['sha256'],scenes=sequences,bytes=total)
    (output/'.archive.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    p.add_argument('--sequence',action='append',required=True);p.add_argument('--spec',default='configs/pandaset_archive.json');a=p.parse_args()
    print(json.dumps(extract(a.archive,a.output,a.sequence,json.loads(Path(a.spec).read_text()))))


if __name__=='__main__':main()
