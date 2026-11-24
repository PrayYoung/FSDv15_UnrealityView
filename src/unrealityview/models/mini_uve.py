import torch
from torch import nn
import torch.nn.functional as F
from unrealityview.geometry import plucker_rays


class MiniUVE(nn.Module):
    """Frozen DINOv2-S/14 + fixed geometry embedding; default 6×32 width384 tokens.

    Geometry projection is deterministic and frozen so cached tokens do not
    silently become stale during downstream DiT/policy training.
    """
    def __init__(self, pretrained=True, backbone=None, token_grid=(4,8)):
        super().__init__()
        self.token_grid=tuple(token_grid)
        if self.token_grid not in ((4,8),(5,8),(10,16)):raise ValueError("Unsupported UVE grid")
        if backbone is None:
            import timm
            backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                         pretrained=pretrained, img_size=(140, 224),
                                         num_classes=0)
        self.backbone = backbone.eval().requires_grad_(False)
        gen = torch.Generator().manual_seed(17)
        self.register_buffer("ray_projection", torch.randn(6, 384, generator=gen) / 6**.5)
        self.register_buffer("mean", torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, cameras, intrinsics=None, camera_to_ego=None, camera_mask=None):
        if cameras.ndim != 5 or tuple(cameras.shape[1:]) != (6,3,140,224):
            raise ValueError("cameras must be (B,6,3,140,224), RGB in [0,1]")
        b = len(cameras)
        x = (cameras.flatten(0,1)-self.mean)/self.std
        feats = self.backbone.forward_features(x)
        if isinstance(feats, dict):
            feats = feats["x_norm_patchtokens"]
        else:
            prefix = getattr(self.backbone, "num_prefix_tokens", 1)
            feats = feats[:, prefix:]
        spatial = feats.reshape(b*6,10,16,384).permute(0,3,1,2)
        h,w=self.token_grid
        if self.token_grid==(10,16):pooled=spatial
        elif self.token_grid==(5,8):pooled=spatial.reshape(b*6,384,5,2,8,2).mean((3,5))
        else:
            columns=spatial.reshape(b*6,384,10,8,2).mean(-1)
            pooled=torch.stack([columns[:,:,i*10//4:((i+1)*10+3)//4].mean(2) for i in range(4)],2)
        tokens=pooled.flatten(2).transpose(1,2).reshape(b,6,h*w,384)
        if (intrinsics is None) != (camera_to_ego is None):
            raise ValueError("Provide intrinsics and camera_to_ego together")
        if intrinsics is not None:
            rays = plucker_rays(intrinsics.float(), camera_to_ego.float(),height=h,width=w).to(tokens)
            tokens = tokens + rays @ self.ray_projection.to(tokens)
        if camera_mask is not None:
            if camera_mask.shape!=(b,6) or not camera_mask.bool().any(-1).all():
                raise ValueError("camera_mask must be B,6 with at least one available camera")
            tokens=tokens*camera_mask.to(tokens)[:,:,None,None]
        return tokens.reshape(b,6*h*w,384)
