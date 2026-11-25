from types import SimpleNamespace
import pytest
from tools.remote_runner.failure_recovery import cleanup_infrastructure_failure


def test_recover_available_outputs_then_record_destruction():
    events=[];updates=[];attempt={};paths={'metadata':'meta','vast_log':'log','outputs':'out'}
    def sync(*args,**kwargs):assert kwargs['transfer_timeout_seconds']==45;events.append('sync');return ['return']
    result=cleanup_infrastructure_failure(7,'connection','work',('return',),paths,'key',attempt,
        fetch_logs=lambda *args:events.append('logs'),sync_outputs=sync,destroy=lambda _:events.append('destroy'),
        update=lambda *args,**kwargs:updates.append(kwargs))
    assert result and events==['logs','sync','destroy'] and attempt['instance_destroyed_at']
    assert updates[-1]['infrastructure_cleanup_confirmed'] is True


def test_unreachable_outputs_do_not_prevent_destruction_and_failure_not_forgotten():
    def failed(*args,**kwargs):raise TimeoutError()
    updates=[];destroyed=[];paths={'metadata':'meta','vast_log':'log','outputs':'out'}
    options=dict(fetch_logs=failed,sync_outputs=failed,destroy=destroyed.append,update=lambda *args,**kwargs:updates.append(kwargs))
    assert cleanup_infrastructure_failure(7,'conn','work',('return',),paths,'key',{},**options)
    assert destroyed==[7]
    options['destroy']=failed
    assert not cleanup_infrastructure_failure(7,None,'',(),paths,'key',{},**options)
    assert updates[-1]['infrastructure_cleanup_error']=='TimeoutError'
