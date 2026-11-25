"""Generate the Arm E result and frozen target checks, then stop for review."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.report_uve_ablation import compare
from unrealityview.data.cache import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--baseline',required=True);a=p.parse_args();root=Path(a.directory)
    current=json.loads((root/'diagnosis/diagnosis.json').read_text());baseline=json.loads(Path(a.baseline).read_text())
    if current['baseline_reproduction']['matched'] is not True:raise ValueError('Validation-best diagnosis did not reproduce its saved evaluation')
    result=compare(baseline,current);new=result.pop('current_10x16');result['current_arm_e']=new
    health=current['splits']['val'].get('cross_attention_health',{})
    stages=health.get('stages',{});health_complete=len(stages)==4
    checks=dict(zero_history_psnr=new['zero_previous']>=19.5,frozen_stage_delta=new['frozen_stage_delta']<=-.15,
        swapped_uve_delta=new['swapped_uve_delta']<=-3.,
        attention_entropy=health_complete and all(s['ent/max']<=.70 for s in stages.values()),
        effective_logit_scale=health_complete and all(s.get('logit_scale_mean',0)>2. for s in stages.values()),
        validation_psnr=new['val_psnr']>21.885,
        frame_decay=new['f7_minus_f0']>result['baseline_4x8']['f7_minus_f0'])
    result.update(checkpoint_step=current['checkpoint_step'],cross_attention_health=health,target_checks=checks,
        conditioning_targets_passed=all(checks[k] for k in ('zero_history_psnr','frozen_stage_delta','swapped_uve_delta','attention_entropy','effective_logit_scale')),
        numeric_quality_targets_passed=checks['validation_psnr'] and checks['frame_decay'],visual_review_required=True,
        treatment='Arm E bundle: cross-only QK scaling,3Dquery init,geometry fusion,historydropout. Existing cached384channels retain their original fixed ray embedding.',
        interpretation='A bundled experiment cannot identify each component causally. OOD controls do not establish a unique root cause.',
        future_training_authorized=False,best_nfe=current['splits']['val']['nfe'])
    result['note']='Existing4x8 cache unchanged. No re-encoding,10x16training,appearancefeatures,CFG or3DGS job.'
    atomic_json(root/'comparison.json',result)
    rows=['# PandaSet v3 Arm E: 5,000-step pilot','',
        'Read the metrics and visuals together. These are research diagnostics, not a finished demo.','',
        '| Measurement | v2 baseline | Arm E |','|---|---:|---:|']
    for key in ('val_psnr','val_pixel_l1','val_latent_l1','train_psnr','train_val_gap','f7_minus_f0','frozen_stage_delta','swapped_uve_delta','zero_previous','swapped_previous'):
        rows.append(f"| {key} | {result['baseline_4x8'][key]:.5f} | {new[key]:.5f} |")
    rows+=['',f"Best checkpoint: step {current['checkpoint_step']}. Validation PSNR change: {result['val_psnr_delta']:+.5f} dB.",
        '',f"Frozen target checks: {json.dumps(checks,sort_keys=True)}",'',
        'Entropy uses log(valid key count), averaged per stage over baseline validation 1-NFE queries/heads. Scale is exp(logit_scale), capped at 8; raw log-parameter means are also recorded.',
        'Conditioning entropy/scale targets must hold in all four stage means. No baseline attention-entropy measurement exists, so no measured pre/post collapse claim is made.',
        'Original cached features already include the old fixed ray projection; Arm E adds learned fusion without rewriting those channels.',
        'No reserved test scenes were uploaded or evaluated. The local JPEG parity mismatch remains unresolved and does not gate this experiment on unchanged cache features.',
        'Multiple architectural/training changes form one bundled treatment; component effects are not isolated. Wrong-scene/zero-history controls remain OOD probes.',
        '', '[Full comparison and attention health](comparison.json)', '[Train/validation diagnosis](diagnosis/diagnosis.json)',
        '', '![Training reconstruction](diagnosis/train_contact_sheet.png)', '', '![Validation reconstruction](diagnosis/val_contact_sheet.png)',
        '', 'A rollout is retained under train/eval_STEP/closed_loop_32_frames.gif for the validation-best step; inspect it before any visual pass.',
        '', 'STOP: no additional training, styles, geometry jobs or automatic follow-up.']
    (root/'README.md').write_text('\n'.join(rows)+'\n')


if __name__=='__main__':main()
