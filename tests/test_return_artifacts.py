import copy
import json
import pytest
import torch
from safetensors.torch import load_file,save_file
from scripts.compact_return_artifacts import compact,export_latest,compact_visuals
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


def test_visual_retention_keeps_initial_best_final_metrics_and_unique_files(tmp_path):
    for name,score in [('eval_000000',8),('eval_000500',20),('eval_001000',22),('eval_001500',21),('eval_raw_001000',23)]:
        path=tmp_path/name;path.mkdir()
        (path/'metrics.json').write_text(json.dumps({'nfe':{'1':{'psnr_db':score}}}))
        (path/'contact_sheet.png').write_bytes(b'preview')
        (path/'closed_loop.gif').write_bytes(b'preview')
        (path/'unique.safetensors').write_bytes(b'learned')
    compact_visuals(tmp_path)
    for path in tmp_path.glob('eval_*'):
        assert (path/'metrics.json').exists() and (path/'unique.safetensors').exists()
        assert (path/'contact_sheet.png').exists()==(path.name in {'eval_000000','eval_001000','eval_001500'})
    assert json.loads((tmp_path/'visual_retention.json').read_text())['omitted_bytes']==28


def test_invalid_metrics_preserve_every_preview(tmp_path):
    for name,score in [('eval_000000',8),('eval_000500',float('nan'))]:
        path=tmp_path/name;path.mkdir()
        (path/'metrics.json').write_text(json.dumps({'nfe':{'1':{'psnr_db':score}}}))
        (path/'contact_sheet.png').write_bytes(b'preview')
    with pytest.raises(ValueError,match='Nonfinite'):compact_visuals(tmp_path)
    assert len(list(tmp_path.glob('*/contact_sheet.png')))==2
