"""Sequential CPU driver: canonical vast-run owns each of three bounded leases.

No cloud APIs, SSH, readiness polling, background launchers or extra lifecycle
owner. Wait for each foreground CLI, validate returns/destruction, publish, then
start the next authorized chunk. Private state/lock prevents duplicate starts.
"""
import argparse,fcntl,json,os,re,shutil,subprocess,tempfile
from pathlib import Path
from datetime import datetime,timezone
from unrealityview.data.cache import atomic_json
from unrealityview.data.flying_cache import load_scene
from tools.chase_3dgs.flying_plan import validate_selection


def now():return datetime.now(timezone.utc).isoformat()


def recorded_terminal(meta):
    # Failure classification and rental destruction are separate facts.
    return bool(meta.get('instance_destroyed_at') and meta.get('status') in
                ('DESTROYED','INFRA_FAILURE','WORKLOAD_FAILURE','ARTIFACT_FAILURE'))


def remaining_budget(campaign,chunk):
    if any(a.get('approx_cost_usd') is None for c in campaign['chunks'] for a in c.get('attempts',[])):
        raise RuntimeError('Previous rental cost is unknown;no new paid job launched')
    spent=sum(float(a['approx_cost_usd']) for c in campaign['chunks'] for a in c.get('attempts',[]) if a.get('approx_cost_usd') is not None)
    own=sum(float(a['approx_cost_usd']) for a in chunk.get('attempts',[]) if a.get('approx_cost_usd') is not None)
    amount=min(2-own,6-spent)
    if amount<=.05:raise RuntimeError('No meaningful budget remains;no paid job launched')
    return round(amount,6)


def can_advance(summary,attempt):
    return bool(attempt.get('classification')=='SUCCESS' and attempt.get('destroyed_verified') and attempt.get('approx_cost_usd') is not None and attempt['approx_cost_usd']<=attempt['budget_usd'] and summary.get('all_numeric_gates_passed') and summary.get('completed_scenes')==4)


def stage(campaign,chunk):
    template=Path(campaign['template_root']);dest=Path(tempfile.mkdtemp(prefix=f"unrealityview-flying-c{chunk['id']}-",dir='/private/tmp'))
    shutil.copytree(template,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','SOURCE_MANIFEST.json','__pycache__'))
    from tools.chase_3dgs.flying_plan import recipe_hash
    if recipe_hash(dest)[0]!=campaign['recipe_hash']:raise RuntimeError('Frozen source template changed;no paid job launched')
    if json.loads((dest/'configs/flying_export_selection.json').read_text())!=campaign['selection']:raise RuntimeError('Frozen selection changed')
    subprocess.run(['git','init','-q',str(dest)],check=True)
    (dest/'inputs').mkdir(exist_ok=True);atomic_json(dest/'inputs/chunk.json',dict(id=chunk['id']))
    # Only completed,verified same-recipe scenes can enter a resumed chunk.
    if chunk.get('result_dir'):
        source=Path(chunk['result_dir'])/'scenes'
        for item in campaign['selection']['chunks'][chunk['id']-1]['scenes']:
            p=source/item['sequence']
            if (p/'manifest.json').exists():
                load_scene(p,campaign['recipe_hash']);shutil.copytree(p,dest/'inputs/resume/scenes'/item['sequence'])
    import hashlib
    hashes={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file() and '.git' not in p.parts}
    atomic_json(dest/'SOURCE_MANIFEST.json',dict(source_commit=campaign['source_commit'],recipe_hash=campaign['recipe_hash'],files=hashes))
    return dest


def terminal_receipt(cli,run_id,log):
    result=subprocess.run([cli,'status',run_id,'--verify-destroyed'],capture_output=True,text=True,timeout=60)
    log.write(result.stdout+result.stderr);log.flush()
    if result.returncode or 'cleanup_verified_absent=true' not in result.stdout:raise RuntimeError('Exact rental destruction not verified;no next chunk')
    path=re.search(r'^metadata=(.+)$',result.stdout,re.MULTILINE)
    cost=re.search(r'^approx_cost_usd=\$([0-9.]+)$',result.stdout,re.MULTILINE)
    if not path or not cost:raise RuntimeError('Missing canonical terminal receipt')
    metadata_path=Path(path[1]);meta=json.loads(metadata_path.read_text())
    if not recorded_terminal(meta):raise RuntimeError('Nonterminal lease state')
    return metadata_path,meta,float(cost[1])


def run(state_path,resume=False):
    from tools.remote_runner.artifact_sync import validate_return
    from tools.chase_3dgs.flying_publish import publish
    state_path=Path(state_path).resolve();cli=shutil.which('vast-run')
    if not cli:raise RuntimeError('Global vast-run unavailable')
    with state_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=json.loads(state_path.read_text());validate_selection(campaign['selection'])
        if [c['id'] for c in campaign['chunks']]!=[1,2,3]:raise ValueError('Invalid campaign chunk order')
        def save():campaign['last_updated_at']=now();atomic_json(state_path,campaign)
        campaign['controller_pid']=os.getpid();campaign['state']='running';save()
        with state_path.with_suffix('.log').open('a',buffering=1) as log:
            try:
                for chunk in campaign['chunks']:
                    campaign['current_chunk']=chunk['id'];save()
                    if chunk['state']=='published':continue
                    if chunk['state']=='gated':raise RuntimeError('Previous negative numeric result;not retrying it')
                    reuse_terminal=False
                    if chunk.get('attempts'):
                        last=chunk['attempts'][-1]
                        # Resume a completed existing CLI result without resubmitting.
                        if last.get('run_id') and not last.get('terminal_processed'):
                            recorded=Path(last['metadata_path']) if last.get('metadata_path') else None
                            if recorded is None or not recorded.exists() or not recorded_terminal(json.loads(recorded.read_text())):
                                raise RuntimeError('Existing canonical run is not recorded destroyed;no duplicate submission or lifecycle polling')
                            metadata_path,meta,cost=terminal_receipt(cli,last['run_id'],log)
                            last.update(metadata_path=str(metadata_path),classification=meta.get('final_classification'),approx_cost_usd=cost,destroyed_verified=True,terminal_processed=True)
                            chunk['result_dir']=str(Path(meta['outputs_dir'])/'outputs/flying_cache_return');reuse_terminal=True
                        elif chunk['state']!='complete' and not resume:raise RuntimeError('Explicit resume required after an interrupted/failed attempt')
                        if chunk['state']=='complete':reuse_terminal=True
                    if not reuse_terminal:
                        budget=remaining_budget(campaign,chunk);root=stage(campaign,chunk)
                        attempt=dict(source_commit=campaign['source_commit'],started_at=now(),budget_usd=budget,staging_root=str(root),terminal_processed=False)
                        chunk.setdefault('attempts',[]).append(attempt);chunk['state']='launching';save()
                        argv=[cli,'run','--project-root',str(root),'--gpu-model','RTX 3090','--min-reliability','0.99','--min-cuda-version','12.4','--preferred-price','0.20','--max-price','0.35','--disk-gb','120','--max-infra-retries','0','--max-cost-usd',str(budget),'--remote-setup-command-file','scripts/flying_export_setup.sh','--sync-path','outputs/flying_cache_return','--command-file','scripts/flying_export_job.sh','--discard-snapshot-after-upload']
                        for host in sorted({102063,47374,78246,173122,*campaign.get('excluded_host_ids',[])}):argv+=['--exclude-host-id',str(host)]
                        env={**os.environ,'VAST_RUN_PROGRESS_PATH':'outputs/flying_cache_return/progress.json'}
                        process=subprocess.Popen(argv,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                        attempt['owner_cli_pid']=process.pid;save()
                        for line in process.stdout:
                            log.write(line);print(line.rstrip(),flush=True)
                            match=re.search(r'run_id=(vast-[A-Za-z0-9-]+)',line)
                            if match and not attempt.get('run_id'):
                                attempt['run_id']=match[1];chunk['state']='running'
                                project=re.search(r'project=([^\s]+)',line)
                                if project:
                                    attempt['project_id']=project[1]
                                    attempt['metadata_path']=str(Path.home()/'.local/share/vast-run/projects'/project[1]/'runs'/match[1]/'metadata.json')
                                save()
                        code=process.wait();attempt['cli_exit_code']=code
                        if not attempt.get('run_id'):raise RuntimeError('Canonical owner did not report a run identity')
                        metadata_path,meta,cost=terminal_receipt(cli,attempt['run_id'],log)
                        attempt.update(metadata_path=str(metadata_path),classification=meta.get('final_classification'),approx_cost_usd=cost,destroyed_verified=True,terminal_processed=True)
                        chunk['result_dir']=str(Path(meta['outputs_dir'])/'outputs/flying_cache_return');save()
                    attempt=chunk['attempts'][-1];result=Path(chunk['result_dir'])
                    # Provisioning can fail before any workload artifact exists.
                    # Successful workloads still require the full return manifest.
                    if attempt['classification']=='SUCCESS' or result.exists():validate_return(result)
                    summary_path=result/'chunk_summary.json'
                    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
                    if summary and summary['recipe_hash']!=campaign['recipe_hash']:raise RuntimeError('Returned recipe mismatch')
                    if summary.get('scenes'):
                        for row in summary['scenes']:
                            if row['complete']:load_scene(result/'scenes'/row['sequence'],campaign['recipe_hash'])
                    total=sum(float(a.get('approx_cost_usd') or 0) for c in campaign['chunks'] for a in c.get('attempts',[]))
                    chunk['state']='complete' if can_advance(summary,attempt) and total<=6 else 'gated' if attempt['classification']=='SUCCESS' else 'failed'
                    chunk['summary']=summary;campaign['total_estimated_cost_usd']=total;save()
                    campaign['publication_commit']=publish(campaign['repo_root'],campaign);save()
                    if chunk['state']!='complete':
                        campaign['state']='needs_review';save();return 1
                    chunk['state']='published';save()
                campaign['state']='await_all_scene_visual_approval';campaign['step3_launched']=False;save();return 0
            except Exception as error:
                campaign['state']='stopped';campaign['error_type']=type(error).__name__;campaign['error']=str(error);save()
                raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    raise SystemExit(run(a.state,a.resume))
