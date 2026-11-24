import argparse
import sys
from unrealityview.styles import validate_style_slug
from unrealityview.data.stylize_cache import build_style_cache,import_teacher


def main():
    if len(sys.argv)>1 and sys.argv[1]=='import-teacher':
        parser=argparse.ArgumentParser(description='Import fingerprint-bound external RGB style teacher')
        parser.add_argument('--data',required=True);parser.add_argument('--teacher',required=True)
        parser.add_argument('--output',required=True);parser.add_argument('--style',type=validate_style_slug,required=True)
        parser.add_argument('--device',default='auto');a=parser.parse_args(sys.argv[2:])
        import_teacher(a.data,a.teacher,a.output,a.style,a.device);return
    p=argparse.ArgumentParser(description="Build temporally stabilized real-data style teachers; preserve base tokens and poses")
    p.add_argument("--data",default="outputs/real_dataset")
    p.add_argument("--output",default="outputs/styles")
    p.add_argument("--style",choices=("comic","cyberpunk","both"),default="both")
    p.add_argument("--device",default="auto")
    a=p.parse_args();build_style_cache(a.data,a.output,("comic","cyberpunk") if a.style=="both" else (a.style,),a.device)


if __name__=="__main__":main()
