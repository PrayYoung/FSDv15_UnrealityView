import copy,json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from unrealityview.explicit3d.project import composite_live_first,pinhole_close_once
from unrealityview.data.flying_cache import FRAME_IDS,CAMERAS,SIZES,encode_images,pack_mask,unpack_mask,write_scene,load_scene,validate_tensors
from tools.chase_3dgs.flying_plan import validate_selection,numeric_gate,resize_geometry,resize_pair
from tools.chase_3dgs.flying_campaign import remaining_budget,can_advance,terminal_receipt


def test_three_timestamp_priority_and_single_snapshot_pinhole_fill():
    a=torch.zeros(3,3,3);b=torch.ones_like(a);c=torch.ones_like(a)*2
    ma=torch.zeros(3,3,dtype=torch.bool);mb=ma.clone();mc=ma.clone();ma[0,0]=True;mb[0,:2]=True;mc[0]=True
    two,m2=composite_live_first(a,ma,b,mb);three,m3=composite_live_first(two,m2,c,mc)
    assert torch.equal(three[0],torch.tensor([[0.,0,0],[1,1,1],[2,2,2]]))
    rgb=torch.arange(75,dtype=torch.float32).reshape(5,5,3);mask=torch.ones(5,5,dtype=torch.bool);mask[2,2]=False
    closed,valid,filled=pinhole_close_once(rgb,mask)
    assert valid.all() and filled.sum()==1 and torch.equal(closed[mask],rgb[mask])
    neighbors=rgb[1:4,1:4].reshape(-1,3);expected=(neighbors.sum(0)-rgb[2,2])/8
    torch.testing.assert_close(closed[2,2],expected)


def test_large_hole_interior_remains_zero_and_fill_does_not_iterate():
    rgb=torch.ones(11,11,3);mask=torch.ones(11,11,dtype=torch.bool);mask[3:8,3:8]=False
    closed,m,filled=pinhole_close_once(rgb,mask)
    assert not m[4:7,4:7].any() and closed[4:7,4:7].sum()==0
    assert filled.sum()==4 # Only eligible concave corners;no flood through new pixels.
    assert torch.equal(closed[mask],rgb[mask])


def test_frozen_selection_and_numeric_gate_flags():
    plan=json.loads(Path('configs/flying_export_selection.json').read_text());validate_selection(plan)
    bad=copy.deepcopy(plan);bad['chunks'][1]['scenes'][0]['sequence']='040'
    with pytest.raises(ValueError):validate_selection(bad)
    bad=copy.deepcopy(plan);bad['chunks'][0]['scenes'][0]['split']='test'
    with pytest.raises(ValueError):validate_selection(bad)
    render=dict(coverage={'B':[dict(ground=.9)]*68})
    q=numeric_gate(dict(heldout_psnr_db=23.,heldout_samples=42),render)
    assert q['passed'] and q['flag_psnr_below24']
    assert not numeric_gate(dict(heldout_psnr_db=21.,heldout_samples=42),render)['passed']
    assert not numeric_gate(dict(heldout_psnr_db=26.,heldout_samples=42),dict(coverage={'B':[dict(ground=.84)]*68}))['passed']


def test_paired_resize_updates_intrinsics_and_keeps_padding_binary():
    k=np.array([[400.,0,320],[0,400,192],[0,0,1]])
    new,g=resize_geometry((640,384),(512,320),k)
    assert g['resized_size']==[512,307] and g['top']==6
    np.testing.assert_allclose(new[:2,2],[256,159.5])
    image=resize_pair(Image.new('L',(640,384),255),g,mask=True)
    assert image.getbbox()==(0,6,512,313) and set(np.unique(np.asarray(image)))=={0,255}
    points=np.array([1.,2,10]);old_uv=(k@points)[:2]/10;new_uv=(new@points)[:2]/10
    np.testing.assert_allclose(new_uv,[old_uv[0]*.8,old_uv[1]*307/384+6])


def cache_fixture():
    n=68;quality=dict(passed=True,heldout_psnr_db=25.,ground_alpha_min=.9)
    meta=dict(frame_indices=FRAME_IDS,camera_names=CAMERAS,split='train',sequence='040',recipe_hash='fixture',rgb_pixel_sha256={},quality=quality)
    tensors=dict(camera_latents=torch.zeros(n,6,4,20,32,dtype=torch.float16),camera_to_world=torch.eye(4).repeat(n,6,1,1),camera_intrinsics=torch.eye(3).repeat(n,6,1,1),camera_timestamps=torch.arange(n,dtype=torch.float64)[:,None].repeat(1,6)/10,target_to_world=torch.eye(4).repeat(n,1,1),timestamps=torch.arange(n,dtype=torch.float64)/10,frame_indices=torch.tensor(FRAME_IDS))
    for key,(w,h) in SIZES.items():
        tensors[f'target_intrinsics_{key}']=torch.eye(3).repeat(n,1,1)
        for role in ('target','rough'):
            tensors[f'{role}_latents_{key}']=torch.zeros(n,4,h//8,w//8,dtype=torch.float16)
            blob,offset,hashes=encode_images([Image.new('RGB',(w,h),(44,85,120))]);name=f'{role}_rgb_{key}'
            tensors[name+'_webp']=blob.repeat(n);tensors[name+'_offsets']=torch.arange(n+1)*len(blob);meta['rgb_pixel_sha256'][name]=hashes*n
        for mask in ('M','observed_M'):tensors[f'{mask}_{key}_packed']=pack_mask(np.ones((n,h,w),dtype=bool))
    return tensors,meta,quality


def test_compact_cache_lossless_roundtrip_and_corruption_rejection(tmp_path):
    t,m,q=cache_fixture();Image.new('RGB',(32,20)).save(tmp_path/'inspection.png')
    write_scene(tmp_path,t,m,q);restored,metadata,quality,receipt=load_scene(tmp_path,'fixture')
    assert torch.equal(restored['camera_latents'],t['camera_latents']) and quality['passed']
    assert unpack_mask(restored['M_512_packed'],512).all()
    with pytest.raises(ValueError,match='recipe'):load_scene(tmp_path,'wrong')
    with (tmp_path/'cache.safetensors.gz').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):load_scene(tmp_path)


def test_rgb_codec_exact_and_invalid_pose_cannot_enter_cache():
    image=Image.fromarray(np.random.default_rng(3).integers(0,256,(21,33,3),dtype=np.uint8))
    blob,offsets,hashes=encode_images([image]);assert len(blob)>0 and len(hashes)==1
    t,m,q=cache_fixture();t['target_to_world'][0,3,3]=0
    with pytest.raises(ValueError,match='homogeneous'):validate_tensors(t,m,verify_rgb=False)


def test_campaign_budget_and_failure_gates_prevent_unauthorized_advance():
    campaign=dict(chunks=[dict(attempts=[dict(approx_cost_usd=1.1)]),dict(attempts=[dict(approx_cost_usd=.4)]),dict(attempts=[])])
    assert remaining_budget(campaign,campaign['chunks'][1])==1.6
    attempt=dict(classification='SUCCESS',destroyed_verified=True,approx_cost_usd=1,budget_usd=2)
    assert can_advance(dict(all_numeric_gates_passed=True,completed_scenes=4),attempt)
    assert not can_advance(dict(all_numeric_gates_passed=False,completed_scenes=4),attempt)
    assert not can_advance(dict(all_numeric_gates_passed=True,completed_scenes=4),{**attempt,'destroyed_verified':False})
    assert not can_advance(dict(all_numeric_gates_passed=True,completed_scenes=4),{**attempt,'approx_cost_usd':2.01})
    with pytest.raises(RuntimeError,match='unknown'):
        remaining_budget(dict(chunks=[dict(attempts=[dict(approx_cost_usd=None)])]),dict(attempts=[]))


@pytest.mark.parametrize('status',['DESTROYED','INFRA_FAILURE'])
def test_terminal_check_uses_only_canonical_status_and_requires_verified_absence(monkeypatch,tmp_path,status):
    from types import SimpleNamespace
    import io
    metadata=tmp_path/'metadata.json';metadata.write_text(json.dumps(dict(status=status,instance_destroyed_at='time')))
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=0,stdout=f'cleanup_verified_absent=true\napprox_cost_usd=$0.8\nmetadata={metadata}\n',stderr='')
    monkeypatch.setattr('tools.chase_3dgs.flying_campaign.subprocess.run',run)
    path,m,cost=terminal_receipt('vast-run','existing-run',io.StringIO())
    assert calls==[['vast-run','status','existing-run','--verify-destroyed']] and cost==.8
    metadata.write_text(json.dumps(dict(status=status,instance_destroyed_at=None)))
    with pytest.raises(RuntimeError,match='Nonterminal'):terminal_receipt('vast-run','existing-run',io.StringIO())


def test_report_publication_never_copies_dataset_or_private_runtime(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from tools.chase_3dgs.flying_publish import publish
    template=tmp_path/'template';(template/'configs').mkdir(parents=True)
    (template/'configs/flying_export_selection.json').write_text(Path('configs/flying_export_selection.json').read_text())
    result=tmp_path/'private_return';scene=result/'scenes/040';scene.mkdir(parents=True)
    Image.new('RGB',(64,40)).save(scene/'inspection.png')
    (scene/'cache.safetensors.gz').write_bytes(b'PRIVATE_DATASET_CONTENT')
    (scene/'metadata.json').write_text('{"private_path":"/Users/private-person/data"}')
    (scene/'manifest.json').write_text(json.dumps(dict(schema='unrealityview.flying.v1',files={})))
    (result/'chunk_summary.json').write_text(json.dumps(dict(scenes=[dict(sequence='040',split='train',tags=['hill'],complete=True,quality=dict(heldout_psnr_db=25.,ground_alpha_min=.9,passed=True))])))
    campaign=dict(template_root=str(template),source_commit='source',recipe_hash='recipe',chunks=[dict(id=1,state='complete',result_dir=str(result),attempts=[dict(run_id='run',classification='SUCCESS',approx_cost_usd=.1,destroyed_verified=True,metadata_path='/Users/private-person/metadata.json')])])
    def output(argv,**kwargs):return 'main\n' if 'branch' in argv else 'commit\n'
    monkeypatch.setattr('tools.chase_3dgs.flying_publish.subprocess.check_output',output)
    monkeypatch.setattr('tools.chase_3dgs.flying_publish.subprocess.run',lambda *args,**kwargs:SimpleNamespace(returncode=0))
    repo=tmp_path/'repo';publish(repo,campaign);report=repo/'docs/results/pandaset-flying-teacher-12scenes'
    assert not list(report.rglob('*.gz'))
    assert 'PRIVATE_DATASET_CONTENT' not in ''.join(p.read_text() for p in report.rglob('*') if p.suffix in ('.md','.json'))
    assert '/Users/' not in ''.join(p.read_text() for p in report.rglob('*') if p.suffix in ('.md','.json'))
