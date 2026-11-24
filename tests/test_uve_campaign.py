import copy
import json
from pathlib import Path
import pytest
from scripts.uve_grid_budget import decide
from scripts.report_uve_ablation import compare


def test_budget_blocks_slow_training_and_requires_unique_success():
    assert decide([{'status':'passed','projected_training_seconds':1000}])['proceed']
    assert not decide([{'status':'passed','projected_training_seconds':12000}])['proceed']
    assert decide([{'status':'out_of_memory'},{'status':'passed','projected_training_seconds':1000}])['proceed']
    with pytest.raises(ValueError):decide([{'status':'out_of_memory'}])
    with pytest.raises(ValueError):decide([{'status':'passed','projected_training_seconds':1}]*2)


def test_report_rejects_changed_window_identity():
    baseline=json.loads(Path('docs/results/pandaset-v2-pilot-5000/diagnosis/diagnosis.json').read_text())
    result=compare(baseline,baseline)
    assert result['val_psnr_delta']==0 and not result['followup_training_authorized']
    altered=copy.deepcopy(baseline);altered['splits']['val']['windows'][0]['start']+=1
    with pytest.raises(ValueError,match='Validation'):compare(baseline,altered)
