"""Fetch immutable external sources; omit the unneeded gated SMPL import."""
import argparse
import json
from pathlib import Path
import subprocess


def checkout(root,name,spec):
    dest=root/name
    if not (dest/'.git').exists():
        dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','init',str(dest)],check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','-C',str(dest),'remote','add','origin',spec['url']],check=True)
    subprocess.run(['git','-C',str(dest),'fetch','--depth','1','origin',spec['commit']],check=True)
    subprocess.run(['git','-C',str(dest),'checkout','--detach',spec['commit']],check=True)
    actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
    if actual!=spec['commit']:raise ValueError('Upstream revision mismatch')
    # GLM and other native dependencies are gitlinks, absent from a plain fetch.
    # Resolve recorded commits, never the submodules' moving remote branches.
    if name in ('gsplat','pytorch3d','nvdiffrast'):
        subprocess.run(['git','-C',str(dest),'submodule','sync','--recursive'],check=True)
        subprocess.run(['git','-C',str(dest),'submodule','update','--init','--recursive','--depth','1'],check=True)
        status=subprocess.check_output(['git','-C',str(dest),'submodule','status','--recursive'],text=True)
        if any(line and line[0]!=' ' for line in status.splitlines()):
            raise ValueError('Upstream submodule revision mismatch')
    if name=='gsplat' and not (dest/'gsplat/cuda/csrc/third_party/glm/glm/glm.hpp').is_file():
        raise ValueError('Pinned gsplat GLM headers missing before native build')
    return dest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.cache/third_party');p.add_argument('--difix',action='store_true');a=p.parse_args()
    pins=json.loads((Path(__file__).parent/'upstream_pins.json').read_text());root=Path(a.root)
    for name in ('drivestudio','pandaset','gsplat','pytorch3d','nvdiffrast')+(('difix',) if a.difix else ()):
        checkout(root,name,pins[name])
    nodes=root/'drivestudio/models/nodes/__init__.py'
    text=nodes.read_text();nodes.write_text(text.replace('from .smpl import SMPLNodes',''))
    # Only import is removed; SMPL model code is not loaded or used. The
    # generated configuration also removes the SMPL node and load_smpl flag.


if __name__=='__main__':main()
