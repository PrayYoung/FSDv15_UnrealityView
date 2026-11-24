"""Explicit evidence gates between separately invoked pipeline stages."""
import argparse
import hashlib
import json
from pathlib import Path
from unrealityview.data.cache import atomic_json


def require_review(report,review):
    report=Path(report);data=json.loads(report.read_text());decision=json.loads(Path(review).read_text())
    if not data.get('numeric_gate_passed'):
        raise ValueError('Numeric gate has not passed')
    if decision.get('visual_review_passed') is not True or decision.get('report_sha256')!=hashlib.sha256(report.read_bytes()).hexdigest():
        raise ValueError('Visual review must pass and bind to the exact numeric report')
    return data


def require_numeric(report):
    data=json.loads(Path(report).read_text())
    if data.get('numeric_gate_passed') is not True:raise ValueError('Numeric gate has not passed')
    return data


def front_gate(directory):
    directory=Path(directory)
    initial=json.loads((directory/'eval_000000/metrics.json').read_text())
    final=json.loads((directory/'eval_005000/metrics.json').read_text())
    a=initial['nfe']['1'];b=final['nfe']['1'];copy=final['prev_chunk_copy']
    passed=(final['split']=='val' and final['samples']==40 and final['target_views']==['native_front']
            and b['psnr_db']>=15 and b['psnr_db']-a['psnr_db']>=3
            and b['pixel_l1']<=a['pixel_l1']*.75 and b['psnr_db']>copy['psnr_db'])
    return dict(numeric_gate_passed=bool(passed),original_gate_retained=True,additional_gate='beat previous-chunk copy PSNR',
        initial=a,final=b,prev_chunk_copy=copy,taesd_roundtrip=final['taesd_roundtrip'],
        target_view='native_front',test_reserved=True,visual_review_required=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    check=sub.add_parser('check');check.add_argument('--report',required=True);check.add_argument('--review',required=True)
    numeric=sub.add_parser('check-numeric');numeric.add_argument('--report',required=True)
    front=sub.add_parser('front-report');front.add_argument('--train',required=True);front.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='check':require_review(a.report,a.review);print('PIPELINE_GATE_OK')
    elif a.command=='check-numeric':require_numeric(a.report);print('NUMERIC_GATE_ONLY_OK; visual approval not granted')
    else:atomic_json(a.output,front_gate(a.train))


if __name__=='__main__':main()
