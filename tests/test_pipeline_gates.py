import hashlib
import json
import pytest
from scripts.pipeline_gates import require_numeric, require_review


def test_numeric_only_does_not_grant_visual_approval(tmp_path):
    report=tmp_path/'gate.json';report.write_text(json.dumps({'numeric_gate_passed':True}))
    review=tmp_path/'review.json';review.write_text(json.dumps({'visual_review_passed':False,
        'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}))
    assert require_numeric(report)['numeric_gate_passed'] is True
    with pytest.raises(ValueError,match='Visual review'):require_review(report,review)


@pytest.mark.parametrize('value',[False,'true',None])
def test_numeric_check_fails_closed(tmp_path,value):
    report=tmp_path/'gate.json';report.write_text(json.dumps({'numeric_gate_passed':value}))
    with pytest.raises(ValueError,match='Numeric gate'):require_numeric(report)
