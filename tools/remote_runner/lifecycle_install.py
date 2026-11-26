"""Install bounded transfer, wake protection and cleanup verification in Vast owner."""
from pathlib import Path
import argparse,shutil


def patch(source):
    if 'from vast_lifecycle import prevent_idle_sleep' in source:return source
    source=source.replace('    project_id = project_identifier(project_root)',
        '    from vast_lifecycle import prevent_idle_sleep\n    prevent_idle_sleep()\n    project_id = project_identifier(project_root)',1)
    start=source.index('    delay_seconds = ',source.index('def schedule_watchdog('));end=source.index('    process = subprocess.Popen(',start)
    source=source[:start]+'''    cmd = watchdog_command(config.bin_path, run_id, int(deadline.timestamp()))
'''+source[end:]
    source=source.replace('["nohup", "sh", "-c", shell_command]','cmd',1)
    old='''    remaining = max(0.0, (deadline - utcnow()).total_seconds())
    if remaining:
        time.sleep(remaining)'''
    new='''    from vast_lifecycle import prevent_idle_sleep, wait_wall_deadline
    prevent_idle_sleep()
    wait_wall_deadline(deadline)'''
    if old not in source:raise ValueError('Watchdog deadline anchor changed')
    source=source.replace(old,new,1)
    source=source.replace('with tarfile.open(fileobj=archive_process.stdout, mode="r|gz") as tar:',
        'from vast_lifecycle import skip_login_banner\n            with tarfile.open(fileobj=skip_login_banner(archive_process.stdout), mode="r|gz") as tar:',1)
    # Apply a hard transfer timeout in the normal finalizer as well as recovery.
    old='''                        paths["outputs"],
                    )'''
    if old not in source:raise ValueError('Final transfer anchor changed')
    source=source.replace(old,'''                        paths["outputs"],
                        transfer_timeout_seconds=120,
                    )''',1)
    source=source.replace('    status_parser.add_argument("run_id")','    status_parser.add_argument("run_id")\n    status_parser.add_argument("--verify-destroyed", action="store_true")',1)
    source=source.replace('    print_run_summary(metadata, live_instance)\n    print(f"metadata={path}")',
        '''    if getattr(args, "verify_destroyed", False):
        if not instance_id:raise InfrastructureError("No recorded instance to verify")
        confirm_instance_gone(int(instance_id))
        print("cleanup_verified_absent=true")
    print_run_summary(metadata, live_instance)
    print(f"metadata={path}")''',1)
    compile(source,'vast_run.py','exec');return source


def install(path):
    path=Path(path);before=path.read_text();after=patch(before)
    if before!=after:
        shutil.copy2(path,path.with_suffix('.before-lifecycle.py'));path.write_text(after)
    shutil.copy2(Path(__file__).with_name('lifecycle_runtime.py'),path.with_name('vast_lifecycle.py'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runner',required=True);a=p.parse_args();install(a.runner)
