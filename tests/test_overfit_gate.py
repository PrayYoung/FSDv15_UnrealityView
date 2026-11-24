import json
from pathlib import Path
import pytest
from scripts.check_pandaset_v2_overfit import check_gate,summarize_gates,preflight_cache


def result(root,name,one,four,tangent=.1):
    folder=root/name;folder.mkdir(parents=True)
    for step,a,b in zip((0,100,500),one,four):
        path=folder/f'overfit_{step:06d}';path.mkdir()
        (path/'metrics.json').write_text(json.dumps({'nfe':{'1':{'psnr_db':a},'4':{'psnr_db':b}}}))
    rows=[dict(step=step,du_dt_rms=tangent,jvp_backend='exact' if name=='meanflow' else 'skipped',flow_loss=.5)
          for step in (100,500)]
    (folder/'train_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return folder


def test_separate_nfe_controls_small_jitter_and_small_derivative(tmp_path):
    result(tmp_path,'meanflow',[8,27.31,27.27],[8,26,27])
    result(tmp_path,'boundary',[8,10,11],[8,27.31,27.27])
    report=summarize_gates(tmp_path)
    assert report['numeric_gate_passed'] and report['derivative_passed']
    assert report['results']['boundary']['nfe_key']=='4'
    assert report['results']['boundary']['curves_by_nfe']['1'][-1]['psnr']==11
    assert report['visual_review_required'] and not report['full_pilot_authorized_by_gate']


@pytest.mark.parametrize('one,tangent', [([8,27,22.28],.1),([8,27,26.4],.1),([8,23,24],0),([8,23,24],11),([8,23,float('nan')],.1)])
def test_real_regression_zero_exploding_and_nonfinite_fail(tmp_path,one,tangent):
    result(tmp_path,'meanflow',one,[8,23,24],tangent)
    result(tmp_path,'boundary',[8,9,10],[8,23,24])
    assert not summarize_gates(tmp_path)['numeric_gate_passed']


def test_peak_regression_prevents_accumulating_small_drops(tmp_path):
    folder=result(tmp_path,'meanflow',[8,27,26.6],[8,23,24])
    # Insert an intermediate point: two individually tolerated drops exceed peak tolerance.
    path=folder/'overfit_000300';path.mkdir();(path/'metrics.json').write_text(json.dumps({'nfe':{'1':{'psnr_db':26.6},'4':{'psnr_db':24}}}))
    p=folder/'overfit_000500/metrics.json';d=json.loads(p.read_text());d['nfe']['1']['psnr_db']=26.2;p.write_text(json.dumps(d))
    report=check_gate(folder)
    assert report['monotonic_psnr_pass'] and not report['peak_regression_pass']


def test_incomplete_result_and_missing_recovery_fail(tmp_path):
    folder=result(tmp_path,'meanflow',[8,23,24],[8,23,24])
    assert not check_gate(folder,expected_steps=1000)['complete']
    with pytest.raises(ValueError,match='No episodes'):preflight_cache(tmp_path/'missing')


def test_module_docstrings_preserved():
    import train_style_lora
    from unrealityview.data import stylize_cache
    assert train_style_lora.__doc__ and stylize_cache.__doc__
