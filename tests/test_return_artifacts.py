import copy
import json
import pytest
import torch
from safetensors.torch import load_file,save_file
from scripts.compact_return_artifacts import compact,export_latest
from unrealityview.checkpoint import save_checkpoint,load_checkpoint
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.driving_policy_head import DrivingPolicyHead
from unrealityview.models.lora import model_fingerprint
from unrealityview.training import save_training_state,load_training_state


def fixture(root):
    model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4,latent_shape=(2,4,6)))
    policy=DrivingPolicyHead(hidden_dim=16,depth=1,num_heads=4)
    for name in ('checkpoint','checkpoint_best'):save_checkpoint(root/name,model,policy,42)
    state=dict(step=42,ema_model=model.state_dict(),ema_policy=policy.state_dict(),
               model=copy.deepcopy(model.state_dict()),optimizer={'moment':torch.ones(3)},rng={'probe':123})
    save_training_state(root/'checkpoint',state)
    return model,policy


def test_compaction_preserves_exact_restart_and_best_and_exports_latest(tmp_path):
    model,policy=fixture(tmp_path);original=model_fingerprint(model)
    compact(tmp_path)
    assert not (tmp_path/'checkpoint/model.safetensors').exists()
    state=load_training_state(tmp_path/'checkpoint')
    assert state['rng']['probe']==123 and torch.equal(state['optimizer']['moment'],torch.ones(3))
    best,_,_=load_checkpoint(tmp_path/'checkpoint_best');assert model_fingerprint(best)==original
    assert json.loads((tmp_path/'return_payload.json').read_text())['restart_complete']
    export_latest(tmp_path);latest,_,_=load_checkpoint(tmp_path/'checkpoint')
    assert model_fingerprint(latest)==original


def test_compaction_never_drops_unique_export(tmp_path):
    fixture(tmp_path);path=tmp_path/'checkpoint/model.safetensors'
    tensors=load_file(str(path));tensors['dit.output.bias'][0]+=1;save_file(tensors,str(path))
    with pytest.raises(ValueError,match='differs'):compact(tmp_path)
    assert path.is_file()
