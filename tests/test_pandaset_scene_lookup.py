import pytest
from tools.chase_3dgs.setup_upstream import normalize_pandaset_scene_lookup


@pytest.mark.parametrize('scene_id',[40,'40','040',2,'002'])
def test_cli_integer_and_sdk_string_ids_reach_same_sequence(tmp_path,scene_id):
    source=tmp_path/'processor.py'
    source.write_text('class Processor:\n    def convert_one(self, scene_idx):\n        scene_data = self.pandaset[scene_idx]\n        scene_data.load()\n        return scene_data\n')
    normalize_pandaset_scene_lookup(source)
    patched=source.read_text();normalize_pandaset_scene_lookup(source)
    assert source.read_text()==patched
    scope={};exec(patched,scope)
    class Sequence:
        loaded=False
        def load(self):self.loaded=True
    sequence=Sequence();processor=scope['Processor']()
    processor.pandaset={str(scene_id).zfill(3):sequence}
    assert processor.convert_one(scene_id) is sequence and sequence.loaded
    with pytest.raises(KeyError):processor.convert_one(999)


def test_changed_upstream_lookup_fails_before_gpu_build(tmp_path):
    path=tmp_path/'processor.py';path.write_text('unexpected upstream source')
    with pytest.raises(ValueError,match='does not match upstream'):
        normalize_pandaset_scene_lookup(path)
