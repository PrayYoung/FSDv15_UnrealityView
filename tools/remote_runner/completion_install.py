"""Expose completed-job recovery through the existing global Vast CLI."""
from pathlib import Path
import argparse,shutil


def patch(source):
    if 'from vast_completion_recovery import recover' in source:return source
    anchor='    destroy_parser = subparsers.add_parser("destroy",'
    at=source.index(anchor)
    source=source[:at]+'''    recovery_parser = subparsers.add_parser("recover", help="Collect a completed existing job after its local owner died; never resubmit.")
    recovery_parser.add_argument("run_id")

'''+source[at:]
    source=source.replace('"run", "status", "reconnect", "destroy", "list-active"','"run", "status", "recover", "reconnect", "destroy", "list-active"')
    source=source.replace('    if args.subcommand == "status":\n        return do_status(args, config)',
        '    if args.subcommand == "status":\n        return do_status(args, config)\n    if args.subcommand == "recover":\n        from vast_completion_recovery import recover\n        return recover(args, config, globals())',1)
    compile(source,'vast_run.py','exec');return source



def patch_orphan(source):
    if 'from vast_orphan_recovery import wait_for_completion' in source:return source
    anchor='    wait_wall_deadline(deadline)'
    replacement='''    from vast_orphan_recovery import wait_for_completion
    path, _ = load_metadata_by_run_id(config, args.run_id)
    if wait_for_completion(deadline, path, config, globals()):
        return 0'''
    if anchor not in source:raise ValueError('Watchdog completion anchor changed')
    source=source.replace(anchor,replacement,1);compile(source,'vast_run.py','exec');return source


def install(path):
    path=Path(path);before=path.read_text();after=patch_orphan(patch(before))
    if before!=after:shutil.copy2(path,path.with_suffix('.before-completion-recovery.py'));path.write_text(after)
    shutil.copy2(Path(__file__).with_name('completion_recovery.py'),path.with_name('vast_completion_recovery.py'))
    shutil.copy2(Path(__file__).with_name('orphan_recovery.py'),path.with_name('vast_orphan_recovery.py'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runner',required=True);a=p.parse_args();install(a.runner)
