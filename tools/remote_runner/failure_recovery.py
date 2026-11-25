"""Bounded artifact recovery and acknowledged cleanup inside the vast-run owner."""
from datetime import datetime,timezone


def cleanup_infrastructure_failure(instance_id,connection,remote_workdir,sync_paths,paths,key,attempt,
                                   *,fetch_logs,sync_outputs,destroy,update):
    # A dead host may have no recoverable outputs. Try once, bound transfers,
    # then prioritize destruction; never silently forget a failed cleanup.
    try:fetch_logs(instance_id,paths['vast_log'])
    except Exception as error:update(paths['metadata'],failure_log_recovery_error=type(error).__name__)
    if connection is not None and remote_workdir:
        try:
            recovered=sync_outputs(connection,key,remote_workdir,sync_paths,paths['outputs'],transfer_timeout_seconds=45)
            update(paths['metadata'],failure_artifact_recovery_attempted=True,synced_paths=recovered)
        except Exception as error:update(paths['metadata'],failure_artifact_recovery_attempted=True,failure_artifact_recovery_error=type(error).__name__)
    try:destroy(instance_id)
    except Exception as error:
        update(paths['metadata'],infrastructure_cleanup_error=type(error).__name__)
        return False
    stamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
    attempt['instance_destroyed_at']=stamp
    update(paths['metadata'],instance_destroyed_at=stamp,infrastructure_cleanup_confirmed=True)
    return True
