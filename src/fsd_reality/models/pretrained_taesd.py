"""TAESD wrapper. INT8 SRAM figures are analytical, execution remains floating point."""
import torch
from torch import nn
from contextlib import nullcontext
from functools import wraps

LATENT_SCALE = 1.0


def precise_convolutions(function):
    """FP32 convolution must not silently become TF32 on CUDA.

    Different tile shapes can select different TF32 cuDNN algorithms. Their
    accumulated errors violate the full-vs-tiled FP32 equivalence contract.
    This does not change explicit FP16/BF16 autocast execution.
    """
    @wraps(function)
    def wrapped(self,tensor,*args,**kwargs):
        context=torch.backends.cudnn.flags(allow_tf32=False) if tensor.device.type=="cuda" else nullcontext()
        with context:return function(self,tensor,*args,**kwargs)
    return wrapped


class PretrainedTAESD(nn.Module):
    def __init__(self, pretrained=True, autoencoder=None):
        super().__init__()
        if autoencoder is None:
            from diffusers import AutoencoderTiny
            autoencoder = (AutoencoderTiny.from_pretrained("madebyollin/taesd", use_safetensors=True,low_cpu_mem_usage=False)
                           if pretrained else AutoencoderTiny())
        self.vae = autoencoder.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    @precise_convolutions
    def encode(self, video):
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("video must be B,3,T,H,W in [-1,1]")
        b, _, t, _, _ = video.shape
        frames = video.permute(0,2,1,3,4).flatten(0,1)
        # Frame-by-frame to keep transient decoder/encoder working sets bounded.
        latents = torch.cat([self.vae.encode(x[None]).latents for x in frames])
        return latents.reshape(b,t,4,*latents.shape[-2:]).permute(0,2,1,3,4).contiguous()

    @torch.no_grad()
    @precise_convolutions
    def decode(self, z):
        if z.ndim != 5 or z.shape[1] != 4:
            raise ValueError("latents must be B,4,T,H,W")
        b, _, t, h, w = z.shape
        frames = z.permute(0,2,1,3,4).reshape(b*t,4,h,w)
        pixels = torch.cat([self.vae.decode(x[None]).sample for x in frames])
        return pixels.reshape(b,t,3,h*8,w*8).permute(0,2,1,3,4).contiguous()

    @torch.no_grad()
    @precise_convolutions
    def decode_tiled(self, z, latent_tile=16, latent_halo=8):
        if latent_tile < 1 or latent_halo < 0:
            raise ValueError("tile must be positive and halo nonnegative")
        b, c, t, h, w = z.shape
        if c != 4:
            raise ValueError("Expected 4 latent channels")
        out = z.new_empty(b,3,t,h*8,w*8)
        for bi in range(b):
            for ti in range(t):
                frame = torch.tanh(z[bi:bi+1,:,ti]/3)*3
                scale = 1
                # Retile BETWEEN decoder modules. A halo of 8 is insufficient
                # for the entire TAESD's ~19-latent receptive radius. Each
                # residual block has radius 3, so module-wise tiling is exact.
                # Full intermediates model off-chip DRAM, not on-chip SRAM.
                for module in self.vae.decoder.layers:
                    if isinstance(module,nn.Upsample):
                        frame = module(frame)
                        scale *= 2
                    elif isinstance(module,nn.ReLU):
                        frame = module(frame)
                    else:
                        radius = 1 if isinstance(module,nn.Conv2d) else 3
                        halo = max(latent_halo,radius)
                        ih,iw = frame.shape[-2:]
                        tile_size = latent_tile*scale
                        assembled = None
                        for y in range(0,ih,tile_size):
                            for x in range(0,iw,tile_size):
                                y1,x1 = min(y+tile_size,ih),min(x+tile_size,iw)
                                ya,xa = max(0,y-halo),max(0,x-halo)
                                yb,xb = min(ih,y1+halo),min(iw,x1+halo)
                                pixels = module(frame[:,:,ya:yb,xa:xb])
                                if assembled is None:
                                    assembled = pixels.new_empty(1,pixels.shape[1],ih,iw)
                                assembled[:,:,y:y1,x:x1] = pixels[:,:,(y-ya):(y1-ya),(x-xa):(x1-xa)]
                        frame = assembled
                out[bi,:,ti] = frame[0]*2-1
        return out

    def sram_budget_report(self, latent_tile=16, latent_halo=8):
        """Conservative fused-tile estimate: INT8 weights, three FP16 feature buffers.

        A hardware implementation must retain only a layer's live buffers and
        stream output pixels to DRAM. PyTorch allocations/workspaces are excluded.
        """
        decoder = self.vae.decoder
        weight_bytes = sum(p.numel() for p in decoder.parameters())
        side = (latent_tile+2*latent_halo)*8
        channels = max((m.out_channels for m in decoder.modules() if isinstance(m,nn.Conv2d)),default=64)
        activation_bytes = 3 * side * side * channels * 2
        total = weight_bytes + activation_bytes
        return {"int8_weight_bytes":weight_bytes,"fp16_tiled_activation_bytes":activation_bytes,
                "total_mib":total/2**20,"budget_mib":32,"within_budget":total < 32*2**20,
                "assumptions":"Analytical fused schedule: 3 FP16 feature buffers + INT8 weights; not measured hardware SRAM",
                "execution_dtype":str(next(decoder.parameters()).dtype),"latent_tile":latent_tile,"latent_halo":latent_halo}
