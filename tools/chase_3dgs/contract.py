"""Pure PNG/JSON export and exact cockpit calibration; no external 3DGS imports."""
import json
import hashlib
import math
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_chase_teacher import chase_calibration
from tools.chase_3dgs.render_policy import GATE_POLICY
from unrealityview.data.cache import atomic_json

CAMERA=dict(camera_back=6.8,camera_height=3.8,pitch_deg=19.,fov=75.)


def processed_front_pose(scene,index):
    """DriveStudio extrinsics are already camera-to-WORLD, not ego-relative.

    Match its loader: inverse(first ego world pose) @ per-frame world camera.
    Multiplying ego_pose[index] by extrinsics would apply motion twice.
    """
    scene=Path(scene)
    origin=np.loadtxt(scene/'ego_pose/000.txt')
    camera_world=np.loadtxt(scene/'extrinsics'/f'{index:03d}_0.txt')
    return np.linalg.inv(origin)@camera_world


def project_world(points,camera_pose,intrinsics):
    points=np.asarray(points)
    cv=(np.linalg.inv(camera_pose)@np.c_[points,np.ones(len(points))].T).T[:,:3]
    pixels=(intrinsics@cv.T).T
    return pixels[:,:2]/pixels[:,2:3]


def export_manifest(directory,sequence,timestamps,quality,teacher='DriveStudio OmniRe dynamic 3DGS',frame_directory='rgb'):
    directory=Path(directory);manifest=directory/'manifest.json'
    if manifest.exists():manifest.unlink() # Never leave a stale passing manifest.
    times=np.asarray(timestamps,dtype=float)
    if len(times)<2 or not np.isfinite(times).all() or not (np.diff(times)>0).all():raise ValueError('Invalid exact native timestamps')
    if not sequence.isdigit() or len(sequence)!=3:raise ValueError('Invalid PandaSet source identity')
    if frame_directory not in ('rgb','rgb_clean'):raise ValueError('Unsupported teacher frame directory')
    psnr=float(quality['heldout_psnr_db']);coverage=float(quality['chase_alpha_coverage_min'])
    ground=float(quality.get('chase_ground_coverage_min',0.))
    passed=bool(quality.get('heldout_samples',0)>0 and math.isfinite(psnr) and psnr>=22 and math.isfinite(coverage) and coverage>=.50 and math.isfinite(ground) and ground>=.85)
    quality={**quality,'numeric_gate_passed':passed,'visual_review_required':True,'gate_policy':GATE_POLICY,
        'thresholds':dict(heldout_psnr_db=22.,whole_frame_alpha=.50,lower_band_alpha=.85,lower_band_start_row=67)}
    atomic_json(directory/'quality.json',quality)
    if not passed:return None
    frames=[]
    for i in range(len(times)):
        relative=f'{frame_directory}/{i:05d}.png';path=directory/relative
        if directory.resolve() not in path.resolve().parents:raise ValueError('Teacher frame escapes output')
        with Image.open(path) as image:
            if image.mode!='RGB' or image.size!=(320,192):raise ValueError('Teacher requires 320x192 RGB PNGs')
            image.verify()
        frames.append(relative)
    payload=dict(version=1,source_id=f'pandaset/{sequence}',target_view='elevated_chase',
        camera=CAMERA,timestamps=times.tolist(),frames=frames,frame_sha256=[hashlib.sha256((directory/f).read_bytes()).hexdigest() for f in frames],teacher=teacher,
        quality=quality,calibration='World-from-OpenCV pose; front-relative FLU offset; native camera timestamps')
    atomic_json(manifest,payload);return payload
