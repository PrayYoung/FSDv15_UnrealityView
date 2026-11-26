import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from tools.chase_3dgs.cleanup_existing import load_inputs,run_cleanup,marker_pixels
from tools.chase_3dgs.contract import chase_calibration
from unrealityview.data.real_world import CV_TO_FLU


def fixture(root):
    (root/'raw').mkdir(parents=True);(root/'reference').mkdir()
    front=np.eye(4);front[:3,:3]=CV_TO_FLU;pose,k=chase_calibration(front)
    calibration=root/'calibration.json';calibration.write_text(json.dumps([dict(camera_to_world=pose.tolist(),intrinsics=k.tolist())]*80))
    frames=[]
    for i in range(80):
        raw=root/'raw'/f'{i:05d}.png';ref=root/'reference'/f'{i:05d}.jpg'
        Image.new('RGB',(320,192),(40+i,50,70)).save(raw);Image.new('RGB',(320,192),(i,80,40)).save(ref)
        frames.append(dict(index=i,timestamp=100+i/10,raw=str(raw.relative_to(root)),reference=str(ref.relative_to(root)),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest()))
    manifest=dict(sequence='040',raw_source_run='fixture',calibration='calibration.json',calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),frames=frames)
    (root/'manifest.json').write_text(json.dumps(manifest));return manifest


def test_standalone_exports_only_cleaned_outputs_and_three_rows(tmp_path):
    root=tmp_path/'input';fixture(root);out=tmp_path/'return'
    def cleaner(raw,refs,pins,progress):
        assert len(raw)==len(refs)==80;progress('model_loaded',dict(seconds=0.))
        return [image.copy() for image in raw]
    timings=run_cleanup(root,out,cleaner,{'difix_model':{},'difix':{}})
    assert len(list((out/'rgb_clean').glob('*.png')))==80 and not (out/'raw').exists()
    assert Image.open(out/'inspection.png').size==(960,660)
    assert marker_pixels().shape==(3,2)
    record=json.loads((out/'run_receipt.json').read_text());assert not record['raw_refit'] and not record['archive_downloaded']
    assert timings['total_seconds']>=0 and (out/'chase_difix.gif').is_file()
    (root/'raw/00000.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='fingerprint'):load_inputs(root)


def test_existing_cleanup_uses_pinned_single_step_reference_api(monkeypatch):
    from tools.chase_3dgs.render_chase_teacher import cleanup_difix
    calls=[]
    class Pipe:
        @classmethod
        def from_pretrained(cls,*args,**kwargs):calls.append((args,kwargs));return cls()
        def to(self,device):assert device=='cuda';return self
        def __call__(self,prompt,**kwargs):
            assert prompt=='remove degradation' and kwargs['num_inference_steps']==1 and kwargs['timesteps']==[199]
            assert kwargs['guidance_scale']==0. and kwargs['ref_image'].size==(320,192)
            return SimpleNamespace(images=[kwargs['image'].copy()])
    monkeypatch.setitem(sys.modules,'pipeline_difix',SimpleNamespace(DifixPipeline=Pipe))
    pins={'difix_model':{'repo':'fixture','revision':'pinned'}};raw=[Image.new('RGB',(320,192))]
    assert len(cleanup_difix(raw,raw,pins))==1 and calls[0][1]['revision']=='pinned'
    with pytest.raises(ValueError,match='count mismatch'):cleanup_difix(raw,[],pins)
