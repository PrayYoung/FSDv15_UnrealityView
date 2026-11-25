"""Reuse only verified camera caches from the same immutable source revision."""
import hashlib
import json
from pathlib import Path
import shutil
from .cache import load_episode,atomic_json
from .splits import stable_split

RAW_CAMERAS=('front_left','front','front_right','rear_left','rear','rear_right','chase')


def recoverable_episodes(root,revision,selections,frames,raw_cameras=RAW_CAMERAS):
    root=Path(root);old=json.loads((root/'split_manifest.json').read_text())
    if old.get('download',{}).get('revision')!=revision:raise ValueError('Recovery source revision mismatch')
    digest=hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()
    wanted={f"pandaset/{e['sequence']}":e['split'] for e in selections};ready={}
    for path in sorted(root.glob('*/*/episode.safetensors')):
        meta=json.loads(path.with_name('metadata.json').read_text());identity=meta.get('source_id')
        if identity not in wanted:continue
        if (meta.get('split')!=wanted[identity] or path.parent.parent.name!=wanted[identity]
            or meta.get('target_view')!='native_front' or meta.get('frames')!=frames
            or meta.get('split_manifest_sha256')!=digest or meta.get('input_size')!=[224,140]
            or meta.get('pixel_size')!=[320,192]):raise ValueError('Recovery cache identity/configuration mismatch')
        if stable_split('pandaset',identity.split('/')[1],old['seed'])!=wanted[identity]:raise ValueError('Recovery scene split mismatch')
        episode=load_episode(path)
        if len(episode['chase_latents'])!=frames:raise ValueError('Recovery tensor count mismatch')
        for camera in raw_cameras:
            if any(not (path.parent/'raw'/camera/f'{i:05d}.jpg').is_file() for i in range(frames)):
                raise ValueError('Recovery RGB cache incomplete')
        ready[identity]=path.parent
    return ready


def seed_recovery(root,output,manifest):
    revision=manifest['download']['revision'];frames=manifest['download']['frames']
    ready=recoverable_episodes(root,revision,manifest['sequences'],frames)
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    for identity,directory in ready.items():
        destination=Path(output)/directory.parent.name/directory.name
        if destination.exists():raise ValueError('Recovery destination already exists')
        shutil.copytree(directory,destination)
        metadata=json.loads((destination/'metadata.json').read_text())
        metadata['recovered_manifest_sha256']=metadata['split_manifest_sha256']
        metadata['split_manifest_sha256']=digest
        metadata['cache_tensor_sha256']=hashlib.sha256((destination/'episode.safetensors').read_bytes()).hexdigest()
        atomic_json(destination/'metadata.json',metadata)
    return len(ready)
