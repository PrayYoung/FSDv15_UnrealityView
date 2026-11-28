"""Actual frozen TAESD APIs and lossless cache codec before expensive fitting."""
import io,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderTiny


def main():
    pins=json.loads(Path('configs/pixart_stateless_pins.json').read_text())['taesd']
    ae,info=AutoencoderTiny.from_pretrained(pins['repo'],revision=pins['revision'],use_safetensors=True,output_loading_info=True)
    if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys'):raise ValueError('TAESD checkpoint/architecture mismatch')
    ae=ae.cuda().eval().requires_grad_(False);shapes=[]
    with torch.inference_mode():
        for w,h in ((256,160),(320,192),(512,320)):
            latent=ae.encode(torch.zeros(1,3,h,w,device='cuda')).latents
            if tuple(latent.shape)!=(1,4,h//8,w//8) or not torch.isfinite(latent).all():raise ValueError('TAESD shape/numeric failure')
            shapes.append(list(latent.shape))
    pixels=np.random.default_rng(42).integers(0,256,(12,20,3),dtype=np.uint8);image=Image.fromarray(pixels);buffer=io.BytesIO();image.save(buffer,format='WEBP',lossless=True,method=4)
    if not np.array_equal(np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')),pixels):raise ValueError('Lossless RGB codec unavailable')
    record=dict(status='FLYING_CACHE_PREFLIGHT_OK',taesd=pins,latent_shapes=shapes,rgb_codec='lossless_webp')
    Path('.cache/flying_cache_preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
