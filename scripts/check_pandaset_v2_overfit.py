"""Two matched real-data overfit probes. No style, teacher generation or full pilot.

A numeric pass is necessary but still requires review of returned images before
starting the full pilot. Cached source scenes and original split assignments stay
unchanged; training-window metrics are explicitly labelled overfit diagnostics.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import statistics
import math
from unrealityview.data.cache import atomic_json
from train_cached_meanflow import train


def check_gate(directory,nfe_key='1',expected_steps=500,tolerance_db=.5):
    if nfe_key not in ('1','4'):raise ValueError('Gate NFE must be 1 or 4')
    curves=[];curves_by_nfe={'1':[],'4':[]}
    for path in sorted(directory.glob('overfit_*/metrics.json')):
        report=json.loads(path.read_text())
        for key in curves_by_nfe:
            curves_by_nfe[key].append(dict(step=int(path.parent.name.split('_')[-1]),psnr=report['nfe'][key]['psnr_db']))
    curves=curves_by_nfe[nfe_key]
    rows=[json.loads(line) for line in (directory/'train_metrics.jsonl').read_text().splitlines()]
    tangents=[r['du_dt_rms'] for r in rows if r['step']>=100 and r['jvp_backend']!='skipped']
    complete=bool(curves and rows and curves[-1]['step']==expected_steps and rows[-1]['step']==expected_steps)
    finite=all(math.isfinite(c['psnr']) for c in curves) and all(math.isfinite(t) for t in tangents)
    peak=-math.inf;drawdown=0.
    for row in curves:
        peak=max(peak,row['psnr']);drawdown=max(drawdown,peak-row['psnr'])
    return dict(curve=curves,curves_by_nfe=curves_by_nfe,nfe_key=nfe_key,
        complete=complete,finite=finite,tolerance_db=tolerance_db,max_drawdown_db=drawdown,
        final_psnr_pass=bool(complete and finite and curves[-1]['psnr']>=22.),
        monotonic_psnr_pass=bool(complete and finite and all(b['psnr']>=a['psnr']-tolerance_db for a,b in zip(curves,curves[1:]))),
        peak_regression_pass=bool(complete and finite and drawdown<=tolerance_db),
        bounded_weighted_loss_pass=bool(rows and all(math.isfinite(r['flow_loss']) and 0<=r['flow_loss']<=1.000001 for r in rows)),
        du_dt_rms_median=statistics.median(tangents) if tangents else None,
        du_dt_rms_max=max(tangents) if tangents else None)


def summarize_gates(output,expected_steps=500):
    results={name:check_gate(output/name,'1' if name=='meanflow' else '4',expected_steps)
             for name in ('meanflow','boundary')}
    passed={name:all(result[key] for key in ('complete','finite','final_psnr_pass',
        'monotonic_psnr_pass','peak_regression_pass','bounded_weighted_loss_pass'))
        for name,result in results.items()}
    fixed=results['meanflow'];median=fixed['du_dt_rms_median'];maximum=fixed['du_dt_rms_max']
    derivative=bool(median is not None and maximum is not None and 0<median<=3 and maximum<=10)
    return dict(gate_policy='overfit-v2-review-2',numeric_gate_passed=all(passed.values()) and derivative,
        meanflow_quality_passed=passed['meanflow'],boundary_control_passed=passed['boundary'],
        derivative_passed=derivative,visual_review_required=True,full_pilot_authorized_by_gate=False,results=results,
        note='Boundary 4-NFE is a diagnostic control, not evidence of one-step quality; manual geometry review still required')


def preflight_cache(root):
    from unrealityview.data.dataset import CachedEpisodeChunkDataset
    result={}
    for split in ('train','val'):
        data=CachedEpisodeChunkDataset(root,split,jitter=False)
        for path,episode in zip(data.paths,data.episodes):
            frames=len(episode['chase_latents'])
            if not all((path.parent/'raw/chase'/f'{i:05d}.jpg').is_file() for i in range(frames)):
                raise ValueError('Recovery cache is missing reference JPEGs')
        result[split]=len(data.paths)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data')
    p.add_argument('--output',default='outputs/pandaset_v2_overfit');p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--regrade-only',action='store_true',help='Write gate_review2.json from existing metrics; never train or overwrite the original gate')
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True);results={}
    if a.regrade_only:
        atomic_json(output/'gate_review2.json',summarize_gates(output,a.steps));return
    if not a.data:p.error('--data is required unless regrading')
    preflight_cache(a.data)
    if a.preflight_only:print('RECOVERY_CACHE_PREFLIGHT_OK',flush=True);return
    original=json.loads(Path('configs/pandaset_pilot_v2.json').read_text())
    for name,ratio in (('meanflow',.75),('boundary',1.)):
        config={**original,'steps':a.steps,'warmup_steps':50,'jvp_warmup_steps':100,
            'fm_ratio':ratio,'eval_every':100,'eval_items':1,'eval_raw_comparison':False,
            'diagnostic_train_windows':1,'train_jitter':False,'diagnostic_train_eval':True}
        path=output/(name+'_config.json');atomic_json(path,config)
        train(SimpleNamespace(config=path,data=a.data,output=output/name,device=a.device,
            steps=None,batch_size=None,evaluate_initial=True,resume=None,init_weights=None))
        # This gate is not a resumable campaign. Keep final/best weights and
        # diagnostics, omit optimizer/RNG tensors from the local return payload.
        for path in (output/name/'checkpoint').glob('training_state*'):
            path.unlink()
    report=summarize_gates(output,a.steps)
    atomic_json(output/'gate.json',report)
    print('OVERFIT_NUMERIC_GATE',report['numeric_gate_passed'],flush=True)


if __name__=='__main__':main()
