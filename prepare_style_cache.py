import argparse
from unrealityview.data.stylize_cache import build_style_cache


def main():
    p=argparse.ArgumentParser(description="Build temporally stabilized real-data style teachers; preserve base tokens and poses")
    p.add_argument("--data",default="outputs/real_dataset")
    p.add_argument("--output",default="outputs/styles")
    p.add_argument("--style",choices=("comic","cyberpunk","both"),default="both")
    p.add_argument("--device",default="auto")
    a=p.parse_args();build_style_cache(a.data,a.output,("comic","cyberpunk") if a.style=="both" else (a.style,),a.device)


if __name__=="__main__":main()
