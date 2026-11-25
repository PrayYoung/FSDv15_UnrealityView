import json
import pytest
from scripts.collect_chase_return import collect


def test_minimal_return_on_quality_failure_excludes_big_data(tmp_path):
    source=tmp_path/'working';destination=tmp_path/'returned'
    for name in ['archive_record.json','stage_times.json','pandaset/040/quality.json','pandaset/040/inspection.png',
                 'pandaset/040/chase_raw.gif','pandaset/040/rgb/00000.png','ingestion_check/manifest.json',
                 'pandaset/040/alpha/00000.png','ingestion_check/train/episode.safetensors','native.zip','checkpoint.pth']:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
    rows=collect(source,destination,'040',1);names={r['path'] for r in rows}
    assert 'pandaset/040/quality.json' in names and 'pandaset/040/rgb/00000.png' in names
    assert all(not p.endswith(('.pth','.zip','.safetensors')) and '/alpha/' not in p for p in names)
    assert json.loads((destination/'collection.json').read_text())['workload_exit_code']==1
    assert not (destination/'ingestion_check/train/episode.safetensors').exists()


def test_return_rejects_symlinks_and_overlapping_directories(tmp_path):
    source=tmp_path/'source';source.mkdir();outside=tmp_path/'outside';outside.write_text('private')
    (source/'archive_record.json').symlink_to(outside)
    with pytest.raises(ValueError,match='artifact path'):collect(source,tmp_path/'return','040')
    with pytest.raises(ValueError,match='Separate'):collect(source,source/'return','040')
