"""Validate actual pinned depth weights/AMP batch sizes before archive download."""
import json
from pathlib import Path
import torch
from unrealityview.explicit3d.preprocess import preprocess_rgb
from tools.chase_3dgs.explicit_proof import load_depth


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.manual_seed(42);model,pins=load_depth();results=[]
    with torch.inference_mode():
        rgb=torch.full((6,384,640,3),.5,device='cuda')
        for size in (518,392):
            torch.cuda.reset_peak_memory_stats();batch=preprocess_rgb(rgb,size)
            with torch.autocast('cuda',dtype=torch.float16):depth=model(batch)
            torch.cuda.synchronize()
            if depth.shape!=(6,*batch.shape[-2:]) or not torch.isfinite(depth).all():raise RuntimeError('FP16 depth batch preflight failed')
            results.append(dict(input_size=size,batch_shape=list(batch.shape),output_dtype=str(depth.dtype),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    record=dict(status='DEPTH_FP16_PREFLIGHT_OK',model=pins,variants=results,meaning='Pinned model synthetic six-camera AMP forward/API/memory check;not real-data quality evidence')
    target=Path('.cache/depth_fp16_preflight.json');target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
