"""Synthetic fixtures test contracts/geometry only; never model-quality evidence."""
import hashlib
import gzip
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
import pytest
import torch
from tools.chase_3dgs.contract import chase_calibration,processed_front_pose,project_world,export_manifest
from tools.chase_3dgs.extract_pandaset_scene import extract,selected_members
from unrealityview.data.real_world import CV_TO_FLU
from unrealityview.geometry import project_waypoints
from scripts.pipeline_gates import require_review
from prepare_real_data import target_frame,cache_manifest


def test_calibration_matches_cockpit_and_upstream_world_poses(tmp_path):
    angle=.4;rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    front=np.eye(4);front[:3,:3]=rotation@CV_TO_FLU;front[:3,3]=[20,40,1.5]
    pose,k=chase_calibration(front);flu=front.copy();flu[:3,:3]=front[:3,:3]@CV_TO_FLU.T
    points=np.array([[15,0,0,1],[25,3,0,1],[40,-2,0,1]])
    world=(flu@points.T).T[:,:3]
    np.testing.assert_allclose(project_world(world,pose,k),project_waypoints(points[:,:2]),atol=1e-6)
    (tmp_path/'ego_pose').mkdir();(tmp_path/'extrinsics').mkdir()
    ego=np.eye(4);ego[:3,3]=[100,200,0]
    np.savetxt(tmp_path/'ego_pose/000.txt',ego);np.savetxt(tmp_path/'extrinsics/001_0.txt',front)
    np.testing.assert_allclose(processed_front_pose(tmp_path,1),np.linalg.inv(ego)@front)
    assert not np.allclose(processed_front_pose(tmp_path,1),ego@front)


def test_manifest_and_core_cache_ingestion(tmp_path):
    from test_real_data import panda_fixture
    native=tmp_path/'native';panda_fixture(native,'040',48)
    targets=tmp_path/'teacher';directory=targets/'pandaset/040';(directory/'rgb').mkdir(parents=True)
    times=[100+i/10 for i in range(48)]
    for i in range(48):Image.new('RGB',(320,192),(30,50,80)).save(directory/'rgb'/f'{i:05d}.png')
    quality=dict(heldout_psnr_db=24.,heldout_samples=12,chase_alpha_coverage_min=.9,chase_ground_coverage_min=.9)
    report=export_manifest(directory,'040',times,quality)
    assert report['target_view']=='elevated_chase' and len(report['frame_sha256'])==48
    entry=dict(dataset='pandaset',sequence='040',root=str(native),split='train')
    target_frame(targets,entry,0,times[0])
    class Encoder:
        def __call__(self,*a,**k):return torch.zeros(1,192,384)
    class Decoder:
        def encode(self,*a,**k):return torch.zeros(1,4,1,24,40)
    cache_manifest(dict(version=1,seed=42,sequences=[entry]),tmp_path/'cache','cpu',target='teacher',targets=targets,encoder=Encoder(),decoder=Decoder())
    meta=json.loads(next((tmp_path/'cache/train').glob('*/metadata.json')).read_text())
    assert meta['target_view']=='elevated_chase'
    # A failed re-export must invalidate any old importer-visible passing manifest.
    assert export_manifest(directory,'040',times,{**quality,'heldout_psnr_db':10}) is None
    assert not (directory/'manifest.json').exists()


def test_external_extractor_whitelist_integrity_and_budget(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tools.chase_3dgs import extract_pandaset_scene
    monkeypatch.setattr(extract_pandaset_scene.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*2**30))
    archive=tmp_path/'data.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('pandaset/040/camera/front_camera/00.jpg',b'jpeg')
        z.writestr('pandaset/040/lidar/00.pkl.gz',gzip.compress(b'opaque native bytes'))
        z.writestr('pandaset/040/annotations/cuboids/00.pkl.gz',gzip.compress(b'opaque annotations'))
        z.writestr('pandaset/040/annotations/semseg/00.pkl.gz',b'excluded')
        z.writestr('pandaset/041/lidar/00.pkl.gz',b'excluded')
    spec=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='budget'):extract(archive,tmp_path/'small',['040'],spec,max_bytes=1)
    extract(archive,tmp_path/'native',['040'],spec)
    assert (tmp_path/'native/040/lidar/00.pkl').read_bytes()==b'opaque native bytes'
    assert not (tmp_path/'native/041').exists() and not (tmp_path/'native/040/annotations/semseg').exists()
    with zipfile.ZipFile(archive,'a') as z:z.writestr('pandaset/040/camera/front_camera/../../escape.jpg',b'bad')
    with zipfile.ZipFile(archive) as z:
        with pytest.raises(ValueError,match='Unsafe'):selected_members(z,'040')


def test_external_extractor_preserves_low_disk_guard(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tools.chase_3dgs import extract_pandaset_scene
    monkeypatch.setattr(extract_pandaset_scene.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    archive=tmp_path/'data.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('pandaset/040/lidar/00.pkl.gz',b'opaque')
    spec=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(OSError,match='scratch space'):extract(archive,tmp_path/'native',['040'],spec)
    assert not list((tmp_path/'native').rglob('*.pkl.gz'))


def test_visual_gate_requires_exact_report_fingerprint(tmp_path):
    report=tmp_path/'quality.json';report.write_text(json.dumps(dict(numeric_gate_passed=True)))
    review=tmp_path/'review.json';review.write_text(json.dumps(dict(visual_review_passed=True,report_sha256='wrong')))
    with pytest.raises(ValueError,match='exact numeric report'):require_review(report,review)
    review.write_text(json.dumps(dict(visual_review_passed=True,report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())))
    assert require_review(report,review)['numeric_gate_passed']


def test_sky_band_passes_but_empty_and_broken_geometry_fail(tmp_path):
    from tools.chase_3dgs.render_policy import alpha_coverages,GATE_POLICY
    directory=tmp_path/'scene';(directory/'rgb').mkdir(parents=True)
    for i in range(2):Image.new('RGB',(320,192)).save(directory/'rgb'/f'{i:05d}.png')
    alpha=np.ones((192,320));alpha[:67]=0
    c=alpha_coverages(alpha)
    assert .5<c['whole']<.85 and c['ground']==1
    q=dict(heldout_psnr_db=24,heldout_samples=2,chase_alpha_coverage_min=c['whole'],chase_ground_coverage_min=c['ground'])
    report=export_manifest(directory,'040',[100.,100.1],q)
    assert report is not None and report['quality']['gate_policy']==GATE_POLICY
    for changed in (dict(chase_alpha_coverage_min=0,chase_ground_coverage_min=0),dict(chase_ground_coverage_min=.84),dict(chase_alpha_coverage_min=.49),dict(heldout_psnr_db=21.9)):
        assert export_manifest(directory,'040',[100.,100.1],{**q,**changed}) is None
    with pytest.raises(ValueError):alpha_coverages(np.full((192,320),float('nan')))


def test_mean_appearance_scope_preserves_heldout_split_and_restores_on_error():
    from types import SimpleNamespace
    from tools.chase_3dgs.render_policy import consistent_novel_appearance
    original=[0,10,20,30,40,50,60,70]
    affine=SimpleNamespace(in_test_set=False);trainer=SimpleNamespace(test_set_indices=original,models={'Affine':affine})
    assert 1 not in trainer.test_set_indices
    with pytest.raises(RuntimeError):
        with consistent_novel_appearance(trainer,80):
            assert all(i in trainer.test_set_indices for i in range(80))
            affine.in_test_set=True
            raise RuntimeError('render failed')
    assert trainer.test_set_indices is original and not affine.in_test_set


def test_low_ram_build_jobs():
    from tools.chase_3dgs.build_env import build_jobs
    assert build_jobs(16*2**30)==2 and build_jobs(24*2**30)==4 and build_jobs(128*2**30)==4
