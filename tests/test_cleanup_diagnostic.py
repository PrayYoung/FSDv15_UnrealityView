import json
from pathlib import Path
from PIL import Image
from tools.chase_3dgs.diagnostic_common import FRAMES,seal_views,SIZE


def test_explicit_comparison_exports_native_examples_mask_and_three_angle_control(tmp_path,monkeypatch):
    from tools.chase_3dgs import render_chase_teacher,cleanup_diagnostic
    work=tmp_path/'work';work.mkdir();output=tmp_path/'return'
    Image.new('RGB',SIZE,'gray').save(work/'source.png')
    Image.new('RGB',SIZE,'white').save(work/'mask.png')
    groups={name:{i:'source.png' for i in FRAMES} for name in ('camera_only_A','camera_only_B','lidar_anchored_A','lidar_anchored_B')}
    masks=work/'camera_only_B_mask';masks.mkdir()
    for index in FRAMES:(masks/f'{index:05d}.png').write_bytes((work/'mask.png').read_bytes())
    orbit={source:{yaw:'source.png' for yaw in (-25,0,25)} for source in ('camera_only','lidar_anchored')}
    seal_views(work,groups,{i:'source.png' for i in FRAMES},list(range(80)),dict(experiment='explicit3d',orbit=orbit))
    def cleaner(raw,refs,pins,*,progress,cuda_timing):
        assert len(raw)==278 and cuda_timing and all(i.size==SIZE for i in refs)
        progress('model_loaded',dict(seconds=0.));progress('frame',dict(completed_frames=len(raw),seconds=1.,cuda_elapsed_ms=1.))
        return [i.copy() for i in raw]
    monkeypatch.setattr(render_chase_teacher,'cleanup_difix',cleaner)
    cleanup_diagnostic.run(work,output)
    assert Image.open(output/'inspection.png').size==(1920,1100)
    assert Image.open(output/'native_samples/camera_only_B_difix/00025.png').size==SIZE
    assert len(list((output/'orbit').glob('*.png')))==12
    assert 'pointermove' in (output/'orbit.html').read_text()
    metrics=json.loads((output/'cleanup_metrics.json').read_text())
    assert metrics['frames_count']==68 and not metrics['visual_approval']
