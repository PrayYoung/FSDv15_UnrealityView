"""Frozen selection, numeric gates and calibrated paired resizing for Step2."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from unrealityview.data.splits import stable_split

INSPECTION_FRAMES=[12,35,55,75]


def validate_selection(selection):
    chunks=selection['chunks']
    if [c['id'] for c in chunks]!=[1,2,3]:raise ValueError('Expected three ordered chunks')
    seen=set()
    for chunk,counts in zip(chunks,((3,1),(4,0),(3,1))):
        scenes=chunk['scenes']
        if len(scenes)!=4 or tuple(sum(s['split']==split for s in scenes) for split in ('train','val'))!=counts:raise ValueError('Invalid chunk split composition')
        for scene in scenes:
            seq=scene['sequence']
            if len(seq)!=3 or not seq.isdigit() or seq in seen:raise ValueError('Invalid/duplicate sequence')
            if scene['split'] not in ('train','val') or stable_split('pandaset',seq,42)!=scene['split']:raise ValueError('Scene is not in its declared allowed split')
            seen.add(seq)
    return selection


def numeric_gate(native_quality,render_metrics):
    psnr=float(native_quality['heldout_psnr_db']);records=render_metrics['coverage']['B']
    ground=min(float(r['ground']) for r in records) if records else 0.
    passed=bool(np.isfinite(psnr) and native_quality['heldout_samples']>0 and psnr>=22 and ground>=.85 and len(records)==68)
    return dict(passed=passed,heldout_psnr_db=psnr,heldout_samples=native_quality['heldout_samples'],ground_alpha_min=ground,psnr_threshold=22.,ground_alpha_threshold=.85,flag_psnr_below24=psnr<24.,visual_approval=False)


def resize_geometry(source_size,target_size,k):
    w,h=source_size;tw,th=target_size;scale=min(tw/w,th/h)
    rw,rh=round(w*scale),round(h*scale);left=(tw-rw)//2;top=(th-rh)//2
    new=np.asarray(k,dtype=np.float64).copy();new[0]*=rw/w;new[1]*=rh/h;new[0,2]+=left;new[1,2]+=top
    return new,dict(source_size=[w,h],target_size=[tw,th],resized_size=[rw,rh],left=left,top=top)


def resize_pair(image,geometry,mask=False):
    tw,th=geometry['target_size'];rw,rh=geometry['resized_size']
    mode='L' if mask else 'RGB';result=Image.new(mode,(tw,th),0)
    result.paste(image.convert(mode).resize((rw,rh),Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS),(geometry['left'],geometry['top']))
    return result


def recipe_hash(root):
    root=Path(root)
    names=['configs/flying_export_selection.json','configs/depth_metric_pins.json','configs/pixart_stateless_pins.json',
        'configs/scene040_diagnostic.yaml','tools/chase_3dgs/upstream_pins.json','configs/difix_runtime_pins.json',
        'src/unrealityview/explicit3d/project.py','src/unrealityview/explicit3d/preprocess.py',
        'src/unrealityview/data/flying_cache.py','tools/chase_3dgs/native_scene.py','tools/chase_3dgs/flying_plan.py',
        'tools/chase_3dgs/flying_features.py','tools/chase_3dgs/flying_chunk.py','tools/chase_3dgs/diagnostic_refit.py',
        'tools/chase_3dgs/render_chase_teacher.py','tools/chase_3dgs/flying_teacher.py',
        'tools/chase_3dgs/render_policy.py','prepare_chase_teacher.py','configs/difix_requirements.txt',
        'configs/pandaset_archive.json','tools/chase_3dgs/diagnostic_common.py',
        'tools/chase_3dgs/explicit_proof.py','src/unrealityview/data/real_world.py',
        'tools/chase_3dgs/setup_upstream.py','tools/chase_3dgs/requirements.txt']
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),hashes
