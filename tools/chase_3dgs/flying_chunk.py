"""One four-scene workload. Infrastructure remains owned by global vast-run."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from unrealityview.data.cache import atomic_json
from unrealityview.data.flying_cache import load_scene
from tools.chase_3dgs.flying_plan import validate_selection,numeric_gate,recipe_hash
from tools.chase_3dgs.extract_pandaset_scene import extract,validate_native_scene


def summarize(output,entries,recipe,chunk_id):
    scenes=[]
    for entry in entries:
        path=output/'scenes'/entry['sequence'];manifest=path/'manifest.json'
        row={**entry,'complete':False}
        if manifest.exists():
            tensors,meta,quality,receipt=load_scene(path,recipe);del tensors
            if (meta['sequence'],meta['split'])!=(entry['sequence'],entry['split']):raise ValueError('Resume cache identity mismatch')
            row.update(complete=True,quality=quality,cache_bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()))
        elif (path/'quality.json').exists():row['quality']=json.loads((path/'quality.json').read_text())
        scenes.append(row)
    done=all(row['complete'] and row['quality']['passed'] for row in scenes)
    result=dict(schema='unrealityview.flying.chunk.v1',chunk=chunk_id,recipe_hash=recipe,scenes=scenes,completed_scenes=sum(row['complete'] for row in scenes),all_numeric_gates_passed=done,needs_low_psnr_review=[row['sequence'] for row in scenes if row.get('quality',{}).get('flag_psnr_below24')],visual_approval=False,training_launched=False)
    atomic_json(output/'chunk_summary.json',result);return result


def run(chunk_id,output):
    root=Path.cwd();selection=validate_selection(json.loads(Path('configs/flying_export_selection.json').read_text()))
    entries=selection['chunks'][chunk_id-1]['scenes'];recipe,files=recipe_hash(root)
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'recipe.json',dict(recipe_hash=recipe,source_files=files,selection=selection))
    # Only a verified complete cache can be resumed. Failed research gates do not
    # silently trigger a new fit or replacement-scene selection.
    resume=Path('inputs/resume/scenes')
    for entry in entries:
        src=resume/entry['sequence'];dst=output/'scenes'/entry['sequence']
        if src.exists() and not dst.exists():
            load_scene(src,recipe);shutil.copytree(src,dst)
    summary=summarize(output,entries,recipe,chunk_id)
    if summary['all_numeric_gates_passed']:return
    archive=Path('.cache/pandaset_archive/pandaset.zip').resolve();native=Path('.cache/flying/native').resolve()
    processed=Path('.cache/flying/processed').resolve();fit=Path('.cache/flying/fit').resolve();work_root=Path('.cache/flying/work').resolve()
    spec=json.loads(Path('configs/pandaset_archive.json').read_text())
    env={**os.environ,'PATH':str(Path.home()/'.local/bin')+os.pathsep+os.environ.get('PATH','')}
    subprocess.run(['uv','venv','--python','3.11','.download-venv'],check=True,env=env)
    subprocess.run(['uv','pip','install','--python','.download-venv/bin/python','-r','configs/archive_download_requirements.txt'],check=True,env=env)
    subprocess.run(['timeout','--signal=TERM','--kill-after=60s','3600','.download-venv/bin/python','scripts/download_pandaset_archive.py','--record',str(output/'archive_record.json')],check=True,env=env)
    archive_record=json.loads((output/'archive_record.json').read_text())
    if archive_record['verified_sha256']!=spec['sha256'] or archive.stat().st_size!=spec['bytes']:raise ValueError('Verified source archive required')
    started=time.monotonic()
    try:
        for entry in entries:
            seq=entry['sequence'];scene_output=output/'scenes'/seq;work=work_root/seq
            if (scene_output/'manifest.json').exists():continue
            if (scene_output/'quality.json').exists() and not json.loads((scene_output/'quality.json').read_text())['passed']:break
            scene_output.mkdir(parents=True,exist_ok=True)
            atomic_json(output/'progress.json',dict(stage='native_extraction',chunk=chunk_id,scene=seq))
            extraction=extract(archive,native,[seq],spec,verified=True);extraction['validated_frames']=validate_native_scene(native,seq)
            atomic_json(scene_output/'extraction.json',extraction)
            atomic_json(output/'progress.json',dict(stage='teacher_fit',chunk=chunk_id,scene=seq))
            training_env={**env,'UNREALITY_PROGRESS_FILE':str(scene_output/'optimizer_progress.json')}
            subprocess.run(['.drivestudio-venv/bin/python','-m','tools.chase_3dgs.flying_teacher','--sequence',seq,'--native',str(native),'--processed',str(processed),'--scratch',str(fit/seq),'--work',str(work),'--output',str(scene_output)],check=True,env=training_env)
            quality=numeric_gate(json.loads((scene_output/'native_quality.json').read_text()),json.loads((scene_output/'render_metrics.json').read_text()))
            atomic_json(scene_output/'quality.json',quality)
            if not quality['passed']:
                atomic_json(output/'progress.json',dict(stage='numeric_gate_rejected',chunk=chunk_id,scene=seq,quality=quality));break
            atomic_json(output/'progress.json',dict(stage='paired_export',chunk=chunk_id,scene=seq))
            subprocess.run(['.venv/bin/python','-m','tools.chase_3dgs.flying_features','--sequence',seq,'--split',entry['split'],'--recipe',recipe,'--native',str(native),'--processed',str(processed),'--work',str(work),'--output',str(scene_output)],check=True,env=env)
            load_scene(scene_output,recipe)
            summarize(output,entries,recipe,chunk_id)
            for path in (native/seq,processed/seq,fit/seq,work):
                if path.exists():shutil.rmtree(path)
        result=summarize(output,entries,recipe,chunk_id)
        atomic_json(output/'progress.json',dict(stage='chunk_completed' if result['all_numeric_gates_passed'] else 'chunk_gated',chunk=chunk_id,completed_scenes=result['completed_scenes']))
    finally:
        if archive.exists():archive.unlink()
        atomic_json(output/'chunk_workload_seconds.json',dict(seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--chunk',type=int,choices=(1,2,3),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.chunk,a.output)
