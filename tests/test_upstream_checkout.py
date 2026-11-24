import subprocess
import pytest
from tools.chase_3dgs.setup_upstream import checkout


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),'-c','user.name=Test','-c','user.email=test@example.invalid',*args],text=True).strip()


def test_checkout_fetches_recorded_glm_submodule(tmp_path,monkeypatch):
    monkeypatch.setenv('GIT_ALLOW_PROTOCOL','file')
    glm=tmp_path/'glm';glm.mkdir();git(glm,'init')
    (glm/'glm').mkdir();(glm/'glm/glm.hpp').write_text('// pinned test header\n')
    git(glm,'add','.');git(glm,'commit','-m','header');pinned=git(glm,'rev-parse','HEAD')
    upstream=tmp_path/'upstream';upstream.mkdir();git(upstream,'init')
    git(upstream,'submodule','add',str(glm),'gsplat/cuda/csrc/third_party/glm')
    git(upstream,'commit','-am','record submodule');commit=git(upstream,'rev-parse','HEAD')
    # A moving child HEAD must not replace the gitlink recorded by the parent.
    (glm/'glm/glm.hpp').write_text('// later header\n');git(glm,'commit','-am','later')
    result=checkout(tmp_path/'sources','gsplat',{'url':str(upstream),'commit':commit})
    child=result/'gsplat/cuda/csrc/third_party/glm'
    assert git(child,'rev-parse','HEAD')==pinned
    assert (child/'glm/glm.hpp').read_text()=='// pinned test header\n'
    (child/'glm/glm.hpp').unlink()
    with pytest.raises(ValueError,match='GLM headers'):checkout(tmp_path/'sources','gsplat',{'url':str(upstream),'commit':commit})
