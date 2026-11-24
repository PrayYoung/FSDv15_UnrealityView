"""Prepare a descriptive comparison; never launch follow-up experiments."""
import argparse
import json
from pathlib import Path
from unrealityview.data.cache import atomic_json


def compare(baseline,current):
    def stats(report):
        train=report['splits']['train']['nfe']['1'];val=report['splits']['val']['nfe']['1']
        controls=report['splits']['val']['conditioning_1nfe']
        return dict(train_psnr=train['psnr_db'],val_psnr=val['psnr_db'],train_val_gap=train['psnr_db']-val['psnr_db'],
            val_pixel_l1=val['pixel_l1'],val_latent_l1=val['latent_l1'],
            frame_psnr=[f['psnr_db'] for f in val['per_frame']],
            f7_minus_f0=val['per_frame'][7]['psnr_db']-val['per_frame'][0]['psnr_db'],
            frozen_stage_delta=controls['frozen_stage_uve']['psnr_db']-val['psnr_db'],
            swapped_uve_delta=controls['swapped_uve']['psnr_db']-val['psnr_db'],
            zero_previous=controls['zero_z_prev']['psnr_db'],swapped_previous=controls['swapped_z_prev']['psnr_db'])
    old=stats(baseline);new=stats(current)
    if [(x['source_id'],x['start']) for x in baseline['splits']['train']['windows']]!=[(x['source_id'],x['start']) for x in current['splits']['train']['windows']]:raise ValueError('Train comparison window selection changed')
    if [(x['source_id'],x['start']) for x in baseline['splits']['val']['windows']]!=[(x['source_id'],x['start']) for x in current['splits']['val']['windows']]:raise ValueError('Validation comparison window selection changed')
    return dict(baseline_4x8=old,current_10x16=new,val_psnr_delta=new['val_psnr']-old['val_psnr'],
        followup_training_authorized=False,visual_review_required=True,
        note='A small or negative gain does not prove DINO lacks appearance. JPEG reuse and any microbatch difference must be disclosed.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--baseline',required=True);a=p.parse_args();root=Path(a.directory)
    current=json.loads((root/'diagnosis/diagnosis.json').read_text())
    if current['baseline_reproduction']['matched'] is not True:raise ValueError('Current best metric reproduction failed')
    result=compare(json.loads(Path(a.baseline).read_text()),current)
    result.update(best_step=current['checkpoint_step'],best_nfe=current['splits']['val']['nfe'],
        parity=json.loads((root/'parity.json').read_text()),preflight=json.loads((root/'preflight_decision.json').read_text()))
    atomic_json(root/'comparison.json',result)
    rows=['# Step 1: unpooled MiniUVE ablation','',
        'Numerical comparison only; inspect the contact sheets before a quality conclusion. No Step 2 training is launched.','',
        '| Measurement | Baseline 4x8 | Unpooled 10x16 |','|---|---:|---:|']
    for key in ('val_psnr','val_pixel_l1','val_latent_l1','train_psnr','train_val_gap','f7_minus_f0','frozen_stage_delta','swapped_uve_delta','zero_previous','swapped_previous'):
        rows.append(f"| {key} | {result['baseline_4x8'][key]:.5f} | {result['current_10x16'][key]:.5f} |")
    rows += ['',f"Best checkpoint: step {result['best_step']}. Validation PSNR change: {result['val_psnr_delta']:+.5f} dB.",'',
        'Camera features were re-encoded from cached quality90 JPEGs; 4x8 parity results are in parity.json. Original caches/checkpoints are preserved.',
        'Preflight records any batch8/accum2 choice; microbatch RNG equivalence is not assumed.',
        'The train/validation sampling distributions differ. Conditioning substitutions are OOD probes, not a unique causal diagnosis.',
        'Keep the reserved test set unused. Review PSNR, frame decay, conditioning controls and both contact sheets together; then stop.',
        '', '[Detailed comparison](comparison.json)', '[Diagnostic report](diagnosis/diagnosis.json)',
        '', '![Train](diagnosis/train_contact_sheet.png)', '', '![Validation](diagnosis/val_contact_sheet.png)', '']
    (root/'README.md').write_text('\n'.join(rows))


if __name__=='__main__':main()
