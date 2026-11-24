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
from unrealityview.data.cache import atomic_json
from train_cached_meanflow import train


def check_gate(directory):
    curves=[]
    for path in sorted(directory.glob('overfit_*/metrics.json')):
        curves.append(dict(step=int(path.parent.name.split('_')[-1]),psnr=json.loads(path.read_text())['nfe']['1']['psnr_db']))
    rows=[json.loads(line) for line in (directory/'train_metrics.jsonl').read_text().splitlines()]
    tangents=[r['du_dt_rms'] for r in rows if r['step']>=100 and r['jvp_backend']!='skipped']
    return dict(curve=curves,final_psnr_pass=curves[-1]['psnr']>=22.,
        monotonic_psnr_pass=all(b['psnr']>=a['psnr'] for a,b in zip(curves,curves[1:])),
        bounded_weighted_loss_pass=all(0<=r['flow_loss']<=1.000001 for r in rows),
        du_dt_rms_median=statistics.median(tangents) if tangents else None,
        du_dt_rms_max=max(tangents) if tangents else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True)
    p.add_argument('--output',default='outputs/pandaset_v2_overfit');p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=500);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True);results={}
    original=json.loads(Path('configs/pandaset_pilot_v2.json').read_text())
    for name,ratio in (('meanflow',.75),('boundary',1.)):
        config={**original,'steps':a.steps,'warmup_steps':50,'jvp_warmup_steps':100,
            'fm_ratio':ratio,'eval_every':100,'eval_items':1,'eval_raw_comparison':False,
            'diagnostic_train_windows':1,'train_jitter':False,'diagnostic_train_eval':True}
        path=output/(name+'_config.json');atomic_json(path,config)
        train(SimpleNamespace(config=path,data=a.data,output=output/name,device=a.device,
            steps=None,batch_size=None,evaluate_initial=True,resume=None,init_weights=None))
        results[name]=check_gate(output/name)
        # This gate is not a resumable campaign. Keep final/best weights and
        # diagnostics, omit optimizer/RNG tensors from the local return payload.
        for path in (output/name/'checkpoint').glob('training_state*'):
            path.unlink()
    fixed=results['meanflow']
    numeric=all(results[name][key] for name in results for key in
        ('final_psnr_pass','monotonic_psnr_pass','bounded_weighted_loss_pass'))
    numeric=numeric and fixed['du_dt_rms_median'] is not None and .5<=fixed['du_dt_rms_median']<=3 and fixed['du_dt_rms_max']<=10
    atomic_json(output/'gate.json',dict(numeric_gate_passed=numeric,visual_review_required=True,
        full_pilot_authorized_by_gate=False,results=results,
        note='Do not launch Pilot v2 until numeric gates and crisp-geometry visual review pass'))
    print('OVERFIT_NUMERIC_GATE',numeric,flush=True)


if __name__=='__main__':main()
