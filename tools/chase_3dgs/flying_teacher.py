"""External trainer entry point for one authorized Step2 scene, poseB only."""
import argparse
from pathlib import Path
from tools.chase_3dgs.diagnostic_refit import run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True)
    for key in ('native','processed','scratch','work','output'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    if len(a.sequence)!=3 or not a.sequence.isdigit():raise ValueError('Invalid sequence')
    run(native=a.native.resolve(),processed=a.processed.resolve(),scratch=a.scratch.resolve(),work=a.work.resolve(),output=a.output.resolve(),upstream=Path('.cache/third_party/drivestudio').resolve(),sequence=a.sequence,pose_names=('B',),psnr_threshold=22.)
    import json
    from tools.chase_3dgs.diagnostic_common import comparison,load_pil
    from tools.chase_3dgs.flying_plan import INSPECTION_FRAMES
    views=json.loads((a.work/'views.json').read_text())
    refs={i:load_pil(a.work/views['references'][str(i)]) for i in INSPECTION_FRAMES}
    raw={i:load_pil(a.work/views['groups']['pose_B'][str(i)]) for i in INSPECTION_FRAMES}
    comparison([refs,raw],['Native front','PoseB raw teacher'],frames=INSPECTION_FRAMES).save(a.output/'raw_inspection.png')
