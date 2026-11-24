"""Explicitly gated batch fitting; never invoked by the single-scene job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from scripts.download_pandaset_archive import verify_archive
from scripts.pipeline_gates import require_review
from tools.chase_3dgs.extract_pandaset_scene import extract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('archive','front-gate','front-review','teacher-gate','teacher-review'):p.add_argument('--'+arg,required=True)
    p.add_argument('--selection',default='configs/pandaset_validated_pilot.json')
    p.add_argument('--output',default='outputs/chase_teacher');p.add_argument('--iters',type=int,default=15000)
    a=p.parse_args();require_review(a.front_gate,a.front_review);require_review(a.teacher_gate,a.teacher_review)
    selection=json.loads(Path(a.selection).read_text());frozen=json.loads(Path('configs/pandaset_validated_pilot.json').read_text())
    if selection!=frozen:raise ValueError('Batch must use the exact frozen 60/8/8 selection')
    spec=json.loads(Path('configs/pandaset_archive.json').read_text());verify_archive(Path(a.archive),spec)
    scratch=Path('.cache/chase_batch').resolve();scratch.mkdir(parents=True,exist_ok=True)
    output=Path(a.output).resolve();python=Path('.drivestudio-venv/bin/python').resolve()
    for entry in selection['sequences']:
        sequence=entry['sequence'];done=output/'pandaset'/sequence/'manifest.json'
        if done.exists():
            data=json.loads(done.read_text());hashes=data.get('frame_sha256',[])
            if (data.get('source_id')==f'pandaset/{sequence}' and len(hashes)==80 and len(data.get('frames',[]))==80 and
                data.get('quality',{}).get('fit_iterations')==a.iters and data['quality'].get('source_archive_sha256')==spec['sha256'] and
                all(hashlib.sha256((done.parent/path).read_bytes()).hexdigest()==digest for path,digest in zip(data['frames'],hashes))):
                continue
            raise ValueError('Existing teacher export failed integrity validation')
        extract(a.archive,scratch/'native',[sequence],spec,verified=True)
        subprocess.run([str(python),'-m','tools.chase_3dgs.render_chase_teacher',
            '--native',str(scratch/'native'),'--processed',str(scratch/'processed'),
            '--scratch',str(scratch/'fit'),'--sequence',sequence,'--output',str(output),'--iters',str(a.iters)],check=True)
        for path in (scratch/'native'/sequence,scratch/'processed'/sequence,scratch/'fit'/sequence):
            if path.exists():shutil.rmtree(path)


if __name__=='__main__':main()
