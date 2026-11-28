"""Publish only compact reports/strips, never the dataset cache or private state."""
import json,shutil,subprocess
from pathlib import Path
from unrealityview.data.cache import atomic_json


def publish(repo,campaign):
    repo=Path(repo);relative=Path('docs/results/pandaset-flying-teacher-12scenes');out=repo/relative;out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((Path(campaign['template_root'])/'configs/flying_export_selection.json').read_text())
    atomic_json(out/'selection.json',selection)
    rows=[];executions=[]
    for chunk in campaign['chunks']:
        if not chunk.get('result_dir'):continue
        root=Path(chunk['result_dir']);summary_path=root/'chunk_summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            for scene in summary['scenes']:
                seq=scene['sequence'];source=root/'scenes'/seq;public=out/'scenes'/seq;public.mkdir(parents=True,exist_ok=True)
                strip=source/'inspection.png'
                if not strip.exists():strip=source/'raw_inspection.png'
                if strip.exists():shutil.copy2(strip,public/'inspection.png')
                row={k:scene[k] for k in ('sequence','split','tags','complete')};row['chunk']=chunk['id'];row['quality']=scene.get('quality');row['cache_bytes']=scene.get('cache_bytes',0)
                if (source/'manifest.json').exists():
                    receipt=json.loads((source/'manifest.json').read_text());atomic_json(public/'cache_receipt.json',receipt)
                if (source/'feature_metrics.json').exists():
                    features=json.loads((source/'feature_metrics.json').read_text());frames=features['frames']
                    row['holes_mean']={stage:{band:sum(f[stage][band] for f in frames)/len(frames) for band in ('whole','lower_band')} for stage in ('single','two_timestamp','three_timestamp','pinhole_closed')}
                    row['all_live_identity_checks_passed']=features['all_live_identity_checks_passed']
                    atomic_json(public/'feature_summary.json',dict(holes=row['holes_mean'],live_identity_verified=row['all_live_identity_checks_passed'],depth_forward_calls=features['depth_forward_calls'],target_frames=68,preroll_frames=10,depth_input_size=392))
                atomic_json(public/'quality.json',scene.get('quality',{}));rows.append(row)
        executions.append(dict(chunk=chunk['id'],state=chunk['state'],attempts=[{k:a.get(k) for k in ('run_id','classification','approx_cost_usd','destroyed_verified','source_commit')} for a in chunk.get('attempts',[])]))
    report=dict(schema='unrealityview.flying.export.report.v1',source_commit=campaign['source_commit'],recipe_hash=campaign['recipe_hash'],planned_scenes=12,completed_scenes=sum(r['complete'] for r in rows),planned_train_val=[10,2],scenes=rows,chunks=executions,total_estimated_rental_cost_usd=sum(float(a.get('approx_cost_usd',0)) for c in campaign['chunks'] for a in c.get('attempts',[])),maximum_total_budget_usd=6,reported_cost_is_provider_invoice=False,visual_approval=False,training_launched=False,dataset_cache_in_git=False)
    atomic_json(out/'summary.json',report)
    lines=['# PandaSet flying teacher / paired explicit3D export','',f"**{report['completed_scenes']}/12 scenes exported.** Three resumable four-scene chunks;10train/2val from the existing split. No reserved scene is opened or listed. Training remains gated by visual approval of all scenes.",'','| Chunk | Scene | Split | Native PSNR | Ground alpha min | Status |','| --- | --- | --- | ---: | ---: | --- |']
    for row in rows:
        q=row.get('quality') or {};status='exported' if row['complete'] else 'incomplete / gated'
        if q.get('flag_psnr_below24'):status+=';PSNR<24 review flag'
        label=f"[{row['sequence']}](scenes/{row['sequence']}/inspection.png)" if (out/'scenes'/row['sequence']/'inspection.png').exists() else row['sequence']
        lines.append(f"| {row['chunk']} | {label} | {row['split']} | {q.get('heldout_psnr_db',0):.3f} | {q.get('ground_alpha_min',0):.4f} | {status} |")
    lines+=['','PoseB teacher:6cams,800kLiDAR initialization,near_randoms10000,20000configured iterations,held-out native PSNR>=22dB (flag<24),minimum lower-band raw alpha>=0.85. Pinned Difix runs at640x384 with directly resized full native front references. Numerical pass does not establish correct novel geometry.','', 'Student inputs use frozen FP16 metric Small at392,one new six-camera depth pass per source frame,causal t>t-5>t-10 projection,then a single3x3thresholded neighbor fill. M is the post-fill mask;observed_M separately records actual projection coverage. Large-hole interiors remain unfilled;one-pixel eligible boundary changes are possible. Preserving current-covered pixels is not universal moving-object ghost rejection.','', 'Each cache contains both target/rough resolutions512x320and320x192,FP16TAESD latents,packed binary masks,lossless paired RGB for pixel/perceptual-loss evaluation,and full native-source camera256x160TAESD features/calibration/timestamps. Cameras are encoded directly from native JPEGs,not224x140recovery-cache images. Shared letterboxing keeps paired targets,rough and masks aligned.','', 'Only this report,compact receipts and inspection strips are in Git. Compact gzip/safetensors dataset caches stay in local runner returns;native archive,fit checkpoints,models and environments are not downloaded or published. Lossless RGB may make the return larger than the original150–300MB estimate;no lossy compression is used to force that estimate.','',f"Estimated rental cost recorded so far: **${report['total_estimated_rental_cost_usd']:.4f}** of the$6aggregate ceiling,$2maximum per chunk. These are estimates,not provider invoices. Each completed chunk must return verified cache manifests and a verified destroyed instance before another starts.",'','[Frozen protocol](../../FLYING_EXPORT_STEP2.md) · [Full summary](summary.json) · [Selection](selection.json)','']
    for row in rows:
        if (out/'scenes'/row['sequence']/'inspection.png').exists():lines += [f"![Scene {row['sequence']}:front,3-ts rough,M,teacher](scenes/{row['sequence']}/inspection.png)",'']
    (out/'README.md').write_text('\n'.join(lines))
    branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
    if branch!='main':raise RuntimeError('Refusing automatic report commit on a changed branch')
    subprocess.run(['git','-C',str(repo),'add',str(relative)],check=True,timeout=30)
    diff=subprocess.run(['git','-C',str(repo),'diff','--cached','--quiet','--',str(relative)])
    if diff.returncode:
        subprocess.run(['git','-C',str(repo),'-c','user.name=UnrealityView contributors','-c','user.email=contributors@unrealityview.invalid','commit','--only','-m',f"Publish flying export progress: {report['completed_scenes']}/12 scenes",'--',str(relative)],check=True,timeout=60)
    subprocess.run(['git','-C',str(repo),'-c','http.postBuffer=52428800','push','origin','main'],check=True,timeout=120)
    return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
