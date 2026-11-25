import os
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]


def toolkit(tmp_path,version='12.4'):
    root=tmp_path/'cuda toolkit';(root/'bin').mkdir(parents=True);(root/'include').mkdir()
    (root/'include/cuda_runtime.h').write_text('// header fixture')
    compiler=root/'bin/nvcc';compiler.write_text('''#!/bin/bash
if [ "$1" = --version ]; then
 echo 'Cuda compilation tools, release VERSION, V12.4.0'
 exit 0
fi
while [ "$#" -gt 0 ]; do
 if [ "$1" = -o ]; then shift; echo object > "$1"; exit 0; fi
 shift
done
exit 1
'''.replace('VERSION',version));compiler.chmod(0o755)
    return root


def test_hidden_toolkit_is_exported_to_child_builds_and_probe(tmp_path):
    root=toolkit(tmp_path);env={**os.environ,'CUDA_HOME':str(root),'CUDA_PATH':'','PATH':'/usr/bin:/bin'}
    result=subprocess.run(['bash','-c','source "$1"; test "$CUDA_HOME" = "$2"; command -v nvcc; bash "$3"; bash "$4"','test',
        str(ROOT/'tools/chase_3dgs/cuda_env.sh'),str(root),str(ROOT/'tools/chase_3dgs/check_build_env.sh'),str(ROOT/'tools/chase_3dgs/compile_cuda_probe.sh')],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert 'CUDA_COMPILER_OBJECT_OK' in result.stdout
    assert str(root/'bin/nvcc') in result.stdout


def test_runtime_only_does_not_satisfy_toolkit_check(tmp_path):
    if any(Path(p).exists() for p in ['/usr/local/cuda','/opt/cuda']) or list(Path('/usr/local').glob('cuda-12.*')):pytest.skip('System has another usable toolkit')
    root=toolkit(tmp_path);(root/'include/cuda_runtime.h').unlink()
    result=subprocess.run(['bash',str(ROOT/'tools/chase_3dgs/check_build_env.sh')],env={**os.environ,'CUDA_HOME':str(root),'CUDA_PATH':'','PATH':'/usr/bin:/bin'},capture_output=True,text=True)
    assert result.returncode==2 and 'No usable CUDA 12 toolkit' in result.stderr
