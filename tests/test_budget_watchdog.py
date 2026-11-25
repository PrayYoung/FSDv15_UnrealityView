from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
import pytest
from tools.remote_runner.budget_install import patch


SOURCE='''
def configure(run_parser):
    run_parser.add_argument("--dry-run-search", action="store_true")
def run(args,resolved):
    try:
        attempt_count = int(resolved["max_infra_retries"]) + 1
        for i in range(1):
            try:
                instance, connection = wait_for_running_instance()
            finally:
                pass
    finally:
        pass
def do_watchdog(args, config):
    path, metadata = load_metadata_by_run_id(config, args.run_id)
    if metadata.get("status") != "DEBUG_IDLE":
        return 0
    deadline_at = parse_iso(metadata.get("debug_deadline_at"))
    if deadline_at and utcnow() < deadline_at:
        return 0
    if metadata.get("instance_id"):
        destroy_instance(metadata["instance_id"])
    return 0
'''


def test_opt_in_budget_watchdog_covers_normal_run_and_preserves_idle_behavior():
    code=patch(SOURCE);assert patch(code)==code
    now=datetime.now(timezone.utc);actions=[]
    metadata={'status':'RUNNING','budget_watchdog_enabled':True,'budget_deadline_at':(now-timedelta(seconds=1)).isoformat(),'instance_id':7}
    scope=dict(load_metadata_by_run_id=lambda *args:('metadata',metadata),parse_iso=lambda x:datetime.fromisoformat(x) if x else None,
        utcnow=lambda:now,destroy_instance=actions.append)
    exec(code,scope);scope['do_watchdog'](SimpleNamespace(run_id='test'),None);assert actions==[7]
    actions.clear();metadata['instance_destroyed_at']=now.isoformat();scope['do_watchdog'](SimpleNamespace(run_id='test'),None);assert not actions
    metadata.clear();metadata.update(status='RUNNING',instance_id=7);scope['do_watchdog'](SimpleNamespace(run_id='test'),None);assert not actions
    metadata.update(status='DEBUG_IDLE',debug_deadline_at=(now-timedelta(seconds=1)).isoformat());scope['do_watchdog'](SimpleNamespace(run_id='test'),None);assert actions==[7]


@pytest.mark.parametrize('cost,retries',[(1.,1),(0.,0),(float('nan'),0)])
def test_invalid_budget_or_retries_rejected_before_any_wait(cost,retries):
    scope={};exec(patch(SOURCE),scope)
    with pytest.raises(SystemExit):scope['run'](SimpleNamespace(max_cost_usd=cost),{'max_infra_retries':retries})
