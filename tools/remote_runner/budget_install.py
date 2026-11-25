"""Opt-in normal-job budget watchdog patch; all lifecycle actions stay in vast-run."""
import argparse
from pathlib import Path
import shutil


def patch(source):
    if 'budget_watchdog_enabled' in source:return source
    parser='    run_parser.add_argument("--dry-run-search", action="store_true")'
    create='                instance, connection = wait_for_running_instance('
    watchdog='''    if metadata.get("status") != "DEBUG_IDLE":
        return 0
    deadline_at = parse_iso(metadata.get("debug_deadline_at"))'''
    if parser not in source or create not in source or watchdog not in source:raise ValueError('Unsupported budget-watchdog runner layout')
    source=source.replace(parser,parser+'\n    run_parser.add_argument("--max-cost-usd", type=float, default=None, help="Optional one-attempt rental budget; owner watchdog reserves 15 percent headroom")',1)
    # Validate before provisioning. Retry ownership remains with runner; a budget
    # experiment uses exactly one paid attempt to avoid resetting the cost clock.
    anchor='        attempt_count = '
    lines=source.splitlines(True)
    for i,line in enumerate(lines):
        if line.startswith(anchor):
            lines.insert(i,'''        if getattr(args, "max_cost_usd", None) is not None:
            if not 0 < args.max_cost_usd < float("inf"):
                raise SystemExit("max-cost-usd must be positive and finite")
            if resolved["max_infra_retries"] != 0:
                raise SystemExit("A cost-capped job requires max-infra-retries=0 (one paid attempt)")
''');break
    else:raise ValueError('Missing pre-provision budget validation anchor')
    source=''.join(lines)
    block='''                if getattr(args, "max_cost_usd", None) is not None:
                    rate = float(selected_offer["dph_total"])
                    budget_deadline = utcnow() + timedelta(seconds=0.85 * args.max_cost_usd / rate * 3600)
                    update_metadata(paths["metadata"], budget_watchdog_enabled=True,
                        max_cost_usd=args.max_cost_usd,
                        budget_deadline_at=budget_deadline.replace(microsecond=0).isoformat().replace("+00:00", "Z"))
                    budget_pid = schedule_watchdog(config, run_id, budget_deadline, paths["run_dir"] / "budget_watchdog.log")
                    update_metadata(paths["metadata"], budget_watchdog_pid=budget_pid)
'''
    source=source.replace(create,block+create,1)
    replacement='''    budget_active = bool(metadata.get("budget_watchdog_enabled")) and metadata.get("status") != "DESTROYED" and not metadata.get("instance_destroyed_at")
    if not budget_active and metadata.get("status") != "DEBUG_IDLE":
        return 0
    deadline_at = parse_iso(metadata.get("budget_deadline_at") if budget_active else metadata.get("debug_deadline_at"))'''
    source=source.replace(watchdog,replacement,1)
    compile(source,'vast_run.py','exec')
    return source


def install(path):
    path=Path(path);source=path.read_text();updated=patch(source)
    if source!=updated:
        backup=path.with_suffix('.before-budget.py')
        if not backup.exists():shutil.copy2(path,backup)
        path.write_text(updated)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runner',required=True);a=p.parse_args();install(a.runner)
