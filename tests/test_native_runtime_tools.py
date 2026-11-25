import json
from pathlib import Path
import pytest
from tools.chase_3dgs.runtime_env import configure_runtime_tools
from tools.chase_3dgs.progress import record_optimizer_progress
from tools.chase_3dgs.setup_upstream import instrument_optimizer_progress


def test_selected_venv_console_tools_survive_interpreter_symlink(tmp_path):
    directory=tmp_path/'venv/bin';directory.mkdir(parents=True)
    (directory/'python').symlink_to('/usr/bin/python3')
    ninja=directory/'ninja';ninja.write_text('#!/bin/sh\necho 1.11.1\n');ninja.chmod(0o755)
    env={'PATH':'/nonexistent'}
    assert configure_runtime_tools(directory/'python',env)==str(ninja)
    assert env['PATH'].split(':')[0]==str(directory)
    ninja.unlink()
    with pytest.raises(RuntimeError,match='Ninja executable'):configure_runtime_tools(directory/'python',{'PATH':'/nonexistent'})


def test_progress_hook_is_idempotent_and_records_after_optimizer(tmp_path,monkeypatch):
    target=tmp_path/'progress.json';monkeypatch.setenv('UNREALITY_PROGRESS_FILE',str(target))
    source=tmp_path/'train.py';source.write_text('def train(trainer,step,loss_dict):\n        trainer.backward(loss_dict)\n')
    instrument_optimizer_progress(source);text=source.read_text();instrument_optimizer_progress(source);assert source.read_text()==text
    events=[]
    class Trainer:
        def backward(self,loss):assert not target.exists();events.append('optimizer_step')
    namespace={};exec(text,namespace);namespace['train'](Trainer(),9,{'rgb':.1})
    assert events==['optimizer_step'] and json.loads(target.read_text())['completed_updates']==10
    with pytest.raises(ValueError,match='Nonfinite'):record_optimizer_progress(100,{'rgb':float('nan')})
    assert json.loads(target.read_text())['completed_updates']==10
