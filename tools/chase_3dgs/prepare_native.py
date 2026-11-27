"""One verified remote archive download/extraction; no native files returned."""
import argparse,json,subprocess,time
from pathlib import Path
from tools.chase_3dgs.extract_pandaset_scene import extract,validate_native_scene
from unrealityview.data.cache import atomic_json


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spec=json.loads(Path('configs/pandaset_archive.json').read_text())
    subprocess.run(['uv','venv','--python','3.11','.download-venv'],check=True)
    subprocess.run(['uv','pip','install','--python','.download-venv/bin/python','-r','configs/archive_download_requirements.txt'],check=True)
    subprocess.run(['timeout','--signal=TERM','--kill-after=60s','3600','.download-venv/bin/python','scripts/download_pandaset_archive.py','--record',str(output/'archive_record.json')],check=True)
    record=json.loads((output/'archive_record.json').read_text());archive=Path('.cache/pandaset_archive/pandaset.zip')
    if record['verified_sha256']!=spec['sha256'] or archive.stat().st_size!=spec['bytes']:raise ValueError('Missing verified source archive receipt')
    start=time.monotonic();result=extract(archive,Path('.cache/chase_native'),['040'],spec,verified=True)
    result['validated_frames']=validate_native_scene('.cache/chase_native','040')
    result['extraction_seconds']=time.monotonic()-start
    atomic_json(output/'extraction.json',result)
    archive.unlink() # Verified native scene retained remotely; reclaim44GB scratch.


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
