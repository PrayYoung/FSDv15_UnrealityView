"""Versioned, lossless compact flying-view cache. No pickle or native tables."""
import gzip,hashlib,io,json,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file,load
from unrealityview.data.cache import atomic_json

FRAME_IDS=list(range(12,80))
CAMERAS=['front_camera','front_left_camera','front_right_camera','left_camera','right_camera','back_camera']
SIZES={'512':(512,320),'320':(320,192)}
MAX_RAW_BYTES=512*2**20


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda:stream.read(2**20),b''):digest.update(piece)
    return digest.hexdigest()


def pack_mask(mask):
    data=np.asarray(mask,dtype=bool)
    return torch.from_numpy(np.packbits(data,axis=-1,bitorder='little').copy())


def unpack_mask(packed,width):
    return torch.from_numpy(np.unpackbits(packed.numpy(),axis=-1,count=width,bitorder='little').copy()).bool()


def encode_images(images):
    blobs=[];offsets=[0];hashes=[]
    for image in images:
        image=image.convert('RGB');buffer=io.BytesIO();image.save(buffer,format='WEBP',lossless=True,method=4)
        data=buffer.getvalue()
        with Image.open(io.BytesIO(data)) as check:
            if not np.array_equal(np.asarray(check.convert('RGB')),np.asarray(image)):raise ValueError('Lossless RGB roundtrip failed')
        blobs.append(data);offsets.append(offsets[-1]+len(data));hashes.append(hashlib.sha256(np.asarray(image).tobytes()).hexdigest())
    return torch.from_numpy(np.frombuffer(b''.join(blobs),dtype=np.uint8).copy()),torch.tensor(offsets,dtype=torch.int64),hashes


def decode_images(tensors,name):
    blob=tensors[name+'_webp'].numpy().tobytes();offsets=tensors[name+'_offsets'].tolist()
    for a,b in zip(offsets,offsets[1:]):
        with Image.open(io.BytesIO(blob[a:b])) as image:yield image.convert('RGB')


def validate_tensors(tensors,metadata,verify_rgb=True):
    n=68
    if metadata['frame_indices']!=FRAME_IDS or metadata['camera_names']!=CAMERAS:raise ValueError('Cache frame/camera identity mismatch')
    if metadata['split'] not in ('train','val'):raise ValueError('Reserved split forbidden')
    from unrealityview.data.splits import stable_split
    if stable_split('pandaset',metadata['sequence'],42)!=metadata['split']:raise ValueError('Frozen split mismatch')
    shapes={'camera_latents':(n,6,4,20,32),'camera_to_world':(n,6,4,4),'camera_intrinsics':(n,6,3,3),
        'camera_timestamps':(n,6),'target_to_world':(n,4,4),'timestamps':(n,),'frame_indices':(n,)}
    for key,(w,h) in SIZES.items():
        for role in ('target','rough'):shapes[f'{role}_latents_{key}']=(n,4,h//8,w//8)
        for mask in ('M','observed_M'):shapes[f'{mask}_{key}_packed']=(n,h,(w+7)//8)
        shapes[f'target_intrinsics_{key}']=(n,3,3)
    for name,shape in shapes.items():
        value=tensors.get(name)
        if value is None or tuple(value.shape)!=shape or not torch.isfinite(value).all():raise ValueError('Invalid cache tensor: '+name)
        if 'latents' in name and value.dtype!=torch.float16:raise ValueError('Expected FP16 latent cache')
        if name.endswith('_packed') and value.dtype!=torch.uint8:raise ValueError('Expected packed boolean mask')
    if tensors['frame_indices'].dtype!=torch.int64 or tensors['frame_indices'].tolist()!=FRAME_IDS:raise ValueError('Tensor frame ordering mismatch')
    if not torch.all(torch.diff(tensors['timestamps'])>0):raise ValueError('Invalid timestamps')
    for name in ('camera_to_world','target_to_world'):
        poses=tensors[name];rot=poses[...,:3,:3]
        if not torch.allclose(poses[...,3,:],poses.new_tensor([0,0,0,1]).expand_as(poses[...,3,:]),atol=1e-5):raise ValueError('Invalid homogeneous pose')
        if not torch.allclose(rot.transpose(-1,-2)@rot,torch.eye(3).expand_as(rot),atol=1e-3) or not torch.all(torch.linalg.det(rot)>.99):raise ValueError('Invalid camera rotation')
    for name in ('camera_intrinsics','target_intrinsics_512','target_intrinsics_320'):
        k=tensors[name]
        if not torch.all(k[...,0,0]>0) or not torch.all(k[...,1,1]>0):raise ValueError('Invalid camera focal length')
    for key,(w,h) in SIZES.items():
        original=unpack_mask(tensors[f'observed_M_{key}_packed'],w);closed=unpack_mask(tensors[f'M_{key}_packed'],w)
        if (original&~closed).any():raise ValueError('Pinhole close removed observed coverage')
        for role in ('target','rough'):
            name=f'{role}_rgb_{key}';blob=tensors[name+'_webp'];offset=tensors[name+'_offsets']
            if blob.dtype!=torch.uint8 or blob.ndim!=1 or offset.dtype!=torch.int64 or offset.shape!=(n+1,) or offset[0]!=0 or offset[-1]!=len(blob) or not torch.all(torch.diff(offset)>0):raise ValueError('Invalid lossless RGB offsets')
            if verify_rgb:
                hashes=metadata['rgb_pixel_sha256'][name]
                if len(hashes)!=n:raise ValueError('RGB checksum count mismatch')
                for image,digest in zip(decode_images(tensors,name),hashes):
                    if image.size!=(w,h) or hashlib.sha256(np.asarray(image).tobytes()).hexdigest()!=digest:raise ValueError('RGB pixel checksum mismatch')
    return True


def write_scene(directory,tensors,metadata,quality):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    validate_tensors(tensors,metadata)
    temporary=directory/'cache.safetensors.tmp';packed=directory/'cache.safetensors.gz'
    save_file({k:v.cpu().contiguous() for k,v in tensors.items()},str(temporary))
    raw_bytes=temporary.stat().st_size;raw_sha=sha(temporary)
    if raw_bytes>MAX_RAW_BYTES:raise ValueError('Unexpectedly large per-scene compact cache')
    with temporary.open('rb') as source,packed.with_suffix('.gz.tmp').open('wb') as destination:
        with gzip.GzipFile(filename='',mode='wb',fileobj=destination,compresslevel=3,mtime=0) as compressed:
            import shutil
            shutil.copyfileobj(source,compressed,2**20)
    os.replace(packed.with_suffix('.gz.tmp'),packed);temporary.unlink()
    atomic_json(directory/'metadata.json',metadata);atomic_json(directory/'quality.json',quality)
    files={name:dict(bytes=(directory/name).stat().st_size,sha256=sha(directory/name)) for name in ('cache.safetensors.gz','metadata.json','quality.json','inspection.png')}
    record=dict(schema='unrealityview.flying.v1',recipe_hash=metadata['recipe_hash'],sequence=metadata['sequence'],split=metadata['split'],frames=68,raw_safetensors_bytes=raw_bytes,raw_safetensors_sha256=raw_sha,files=files)
    atomic_json(directory/'manifest.json',record)
    load_scene(directory,metadata['recipe_hash']);return record


def load_scene(directory,recipe_hash=None,verify_rgb=True):
    directory=Path(directory);record=json.loads((directory/'manifest.json').read_text())
    if record.get('schema')!='unrealityview.flying.v1' or (recipe_hash and record['recipe_hash']!=recipe_hash):raise ValueError('Cache recipe/schema mismatch')
    if record.get('frames')!=68 or set(record['files'])!={'cache.safetensors.gz','metadata.json','quality.json','inspection.png'}:raise ValueError('Invalid cache manifest file set')
    for name,info in record['files'].items():
        if Path(name).name!=name:raise ValueError('Unsafe cache path')
        p=directory/name
        if p.stat().st_size!=info['bytes'] or sha(p)!=info['sha256']:raise ValueError('Cache file checksum mismatch')
    if not 0<record['raw_safetensors_bytes']<=MAX_RAW_BYTES:raise ValueError('Unsafe decompressed cache size')
    with gzip.open(directory/'cache.safetensors.gz','rb') as stream:raw=stream.read(MAX_RAW_BYTES+1)
    if len(raw)!=record['raw_safetensors_bytes'] or hashlib.sha256(raw).hexdigest()!=record['raw_safetensors_sha256']:raise ValueError('Decompressed cache checksum mismatch')
    metadata=json.loads((directory/'metadata.json').read_text());quality=json.loads((directory/'quality.json').read_text())
    if metadata['quality']!=quality:raise ValueError('Cache quality record mismatch')
    if (metadata['sequence'],metadata['split'],metadata['recipe_hash'])!=(record['sequence'],record['split'],record['recipe_hash']):raise ValueError('Cache identity mismatch')
    tensors=load(raw);validate_tensors(tensors,metadata,verify_rgb)
    return tensors,metadata,quality,record
