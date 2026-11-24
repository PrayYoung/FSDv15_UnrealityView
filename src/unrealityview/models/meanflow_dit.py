from dataclasses import asdict, dataclass
from collections.abc import Callable
import torch
from torch import nn
from .blocks import AdaLNBlock, Attention, TimeEmbed


@dataclass
class DiTConfig:
    latent_channels: int = 4
    latent_shape: tuple[int,int,int] = (8,24,40)
    patch_size: tuple[int,int,int] = (2,2,2)
    hidden_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    context_dim: int = 384
    stages: int = 4
    cross_attention: str = "block"
    time_scale: float = 10.0

    def __post_init__(self):
        if self.cross_attention not in ("block","stage"):raise ValueError("Unknown cross-attention architecture")
        self.latent_shape = tuple(self.latent_shape)
        self.patch_size = tuple(self.patch_size)
        if self.stages != 4 or self.depth % self.stages or self.hidden_dim % self.num_heads:
            raise ValueError("Use four equal stages and hidden_dim divisible by num_heads")
        if any(s % p for s,p in zip(self.latent_shape,self.patch_size)):
            raise ValueError("Latent shape must be divisible by patch size")

    def to_dict(self):
        return asdict(self)


class ConditionalMeanFlowDiT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or DiTConfig()
        self.grid = tuple(s//p for s,p in zip(c.latent_shape,c.patch_size))
        self.patch = nn.Conv3d(c.latent_channels*2,c.hidden_dim,c.patch_size,c.patch_size)
        self.position = nn.Parameter(torch.randn(1,self.grid[0]*self.grid[1]*self.grid[2],c.hidden_dim)*.02)
        self.time = TimeEmbed(c.hidden_dim,scale=c.time_scale)
        self.interval = TimeEmbed(c.hidden_dim,scale=c.time_scale)
        self.blocks = nn.ModuleList([AdaLNBlock(c.hidden_dim,c.num_heads,
                             c.context_dim if c.cross_attention=="block" else None) for _ in range(c.depth)])
        if c.cross_attention=="block":
            self.age_embed=TimeEmbed(c.context_dim,scale=c.time_scale)
        else:
            # Version-1 checkpoints remain readable with their original architecture.
            self.age = TimeEmbed(c.hidden_dim,scale=c.time_scale)
            self.cross = nn.ModuleList([Attention(c.hidden_dim,c.num_heads,c.context_dim) for _ in range(c.stages)])
            self.cross_norm = nn.ModuleList([nn.LayerNorm(c.hidden_dim) for _ in range(c.stages)])
        self.final_norm = nn.LayerNorm(c.hidden_dim,elementwise_affine=False)
        # Non-overlapping ConvTranspose3d is exactly a per-token linear
        # projection + unpatchify. This form also supports Apple MPS.
        self.output = nn.Linear(c.hidden_dim,c.latent_channels*c.patch_size[0]*c.patch_size[1]*c.patch_size[2])
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, z, r, t, z_prev, uve_tokens, stage_ages=None,
                uve_provider: Callable | None = None):
        c = self.config
        if tuple(z.shape[1:]) != (c.latent_channels,*c.latent_shape) or z_prev.shape != z.shape:
            raise ValueError("Latent and previous chunk shapes do not match model config")
        x = self.patch(torch.cat((z,z_prev),1)).flatten(2).transpose(1,2)+self.position
        cond = self.time(t)+self.interval(t-r)
        for stage in range(c.stages):
            tokens = uve_tokens[:,stage] if uve_tokens.ndim == 4 else uve_tokens
            age = torch.zeros_like(t) if stage_ages is None else stage_ages[:,stage]
            if uve_provider is not None:
                update = uve_provider(stage)
                if update is not None:
                    tokens, age = update
            if c.cross_attention=="stage":
                x = x + self.cross[stage](self.cross_norm[stage](x),tokens)
                stage_cond = cond + self.age(age)
                for block in self.blocks[stage*(c.depth//4):(stage+1)*(c.depth//4)]:
                    x = block(x,stage_cond)
            else:
                # Derive the missing-camera mask BEFORE adding freshness embeddings.
                mask=tokens.detach().ne(0).any(-1)
                fresh=tokens+self.age_embed(age)[:,None,:].to(tokens.dtype)
                fresh=fresh*mask[...,None].to(fresh.dtype)
                for block in self.blocks[stage*(c.depth//4):(stage+1)*(c.depth//4)]:
                    x=block(x,cond,fresh,mask)
        x = self.output(self.final_norm(x))
        gt,gh,gw = self.grid
        pt,ph,pw = c.patch_size
        x = x.reshape(len(z),gt,gh,gw,c.latent_channels,pt,ph,pw)
        return x.permute(0,4,1,5,2,6,3,7).reshape(len(z),c.latent_channels,*c.latent_shape)

    @torch.no_grad()
    def sample_chunk(self, z_prev, uve_tokens, num_steps=1, stage_ages=None,
                     uve_provider=None, noise=None, generator=None):
        if num_steps not in (1,2,4):
            raise ValueError("num_steps must be 1, 2, or 4")
        z = (torch.randn(z_prev.shape,device=z_prev.device,dtype=z_prev.dtype,generator=generator)
             if noise is None else noise.clone())
        for i in range(num_steps):
            t = z.new_full((len(z),),1-i/num_steps)
            r = z.new_full((len(z),),1-(i+1)/num_steps)
            z = z-(t-r).view(-1,1,1,1,1)*self(z,r,t,z_prev,uve_tokens,stage_ages,uve_provider)
        return z
