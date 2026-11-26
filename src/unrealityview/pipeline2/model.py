"""Stateless 1-NFE, 240 target + 960 camera tokens, frozen PixArt + LoRA."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from unrealityview.models.lora import LoRALinear


def fourier_rays(rays,bands=4):
    frequencies=2**torch.arange(bands,device=rays.device,dtype=rays.dtype)*math.pi
    angles=rays[...,None]*frequencies
    return torch.cat((rays,angles.sin().flatten(-2),angles.cos().flatten(-2)),-1)


class StatelessPixArt(nn.Module):
    def __init__(self,backbone,rank=16,alpha=32.,checkpoint_blocks=True):
        super().__init__();self.backbone=backbone.requires_grad_(False)
        dimension=backbone.config.num_attention_heads*backbone.config.attention_head_dim
        self.checkpoint_blocks=checkpoint_blocks
        self.role_emb=nn.Embedding(7,dimension)
        self.ray_mlp=nn.Sequential(nn.Linear(54,256),nn.SiLU(),nn.Linear(256,dimension))
        nn.init.zeros_(self.ray_mlp[-1].weight);nn.init.zeros_(self.ray_mlp[-1].bias)
        self.null_caption=nn.Parameter(torch.zeros(1,1,backbone.config.caption_channels))
        for block in backbone.transformer_blocks:
            for name in ('attn1.to_q','attn1.to_k','attn1.to_v','attn1.to_out.0','ff.net.0.proj','ff.net.2'):
                parent,_,leaf=name.rpartition('.');owner=block.get_submodule(parent)
                setattr(owner,leaf,LoRALinear(owner.get_submodule(leaf),rank,alpha))
        for name in ('proj_out','adaln_single'):
            backbone.get_submodule(name).requires_grad_(True)
        backbone.scale_shift_table.requires_grad_(True)

    def forward(self,noise,cameras,camera_rays,target_rays):
        b,c,h,w=noise.shape
        if cameras.shape[1:]!=(6,4,20,32) or noise.shape[1:]!=(4,24,40):
            raise ValueError('Expected six20x32camera latents and24x40target latent')
        base=self.backbone
        targets=base.pos_embed(noise)
        camera_tokens=base.pos_embed(cameras.flatten(0,1)).reshape(b,6,160,-1)
        roles=self.role_emb(torch.arange(6,device=noise.device))[None,:,None]
        camera_tokens=camera_tokens+roles.to(camera_tokens)+self.ray_mlp(fourier_rays(camera_rays).to(self.ray_mlp[0].weight.dtype)).to(camera_tokens)
        targets=targets+self.role_emb.weight[6].to(targets)+self.ray_mlp(fourier_rays(target_rays).to(self.ray_mlp[0].weight.dtype)).to(targets)
        hidden=torch.cat((targets,camera_tokens.flatten(1,2)),1)
        if hidden.shape[1]!=1200:raise ValueError('Stateless contract requires1200tokens')
        # normalized flow t=1 is mapped to the pretrained timestep domain.
        timestep=torch.full((b,),999,device=noise.device,dtype=torch.long)
        cond={'resolution':noise.new_tensor([[192,320]]).expand(b,-1),
              'aspect_ratio':noise.new_full((b,1),192/320)}
        modulation,embedded=base.adaln_single(timestep,cond,batch_size=b,hidden_dtype=hidden.dtype)
        caption=base.caption_projection(self.null_caption.expand(b,-1,-1))
        for block in base.transformer_blocks:
            def apply(value,block=block):
                return block(value,attention_mask=None,encoder_hidden_states=caption,
                             encoder_attention_mask=None,timestep=modulation,
                             cross_attention_kwargs=None,class_labels=None)
            hidden=checkpoint(apply,hidden,use_reentrant=False) if self.training and self.checkpoint_blocks else apply(hidden)
        hidden=hidden[:,:240]
        shift,scale=(base.scale_shift_table[None]+embedded[:,None]).chunk(2,dim=1)
        hidden=base.norm_out(hidden)*(1+scale)+shift
        output=base.proj_out(hidden).reshape(b,12,20,2,2,base.out_channels)
        velocity=torch.einsum('nhwpqc->nchpwq',output).reshape(b,base.out_channels,24,40)[:,:4]
        return noise-velocity

    def adapter_state(self):
        trainable={name for name,p in self.named_parameters() if p.requires_grad}
        return {name:p.detach().float().cpu().contiguous() for name,p in self.named_parameters() if name in trainable}

    def fuse_lora(self):
        for name,module in list(self.named_modules()):
            if isinstance(module,LoRALinear):
                base=module.base
                with torch.no_grad():base.weight.add_((module.lora_b@module.lora_a).to(base.weight)*module.scale.to(base.weight))
                parent,_,leaf=name.rpartition('.');setattr(self.get_submodule(parent),leaf,base)
        return self
