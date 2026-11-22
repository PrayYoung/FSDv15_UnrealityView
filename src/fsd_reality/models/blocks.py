import math
import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64,dim),nn.SiLU(),nn.Linear(dim,dim))

    def forward(self, t):
        freq = torch.exp(-math.log(10000)*torch.arange(32,device=t.device,dtype=torch.float32)/31)
        phase = t.float().reshape(-1,1)*freq[None]*1000
        return self.net(torch.cat((phase.cos(),phase.sin()),-1).to(self.net[0].weight.dtype))


class Attention(nn.Module):
    def __init__(self, dim, heads, context_dim=None):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim,dim)
        self.kv = nn.Linear(context_dim or dim,dim*2)
        self.out = nn.Linear(dim,dim)

    def forward(self, x, context=None):
        context = x if context is None else context
        b,n,d = x.shape
        q = self.q(x).reshape(b,n,self.heads,d//self.heads).transpose(1,2)
        k,v = self.kv(context).reshape(b,-1,2,self.heads,d//self.heads).permute(2,0,3,1,4)
        # RMS rather than L2 norm preserves the standard 1/sqrt(head_dim) scale.
        q = q * (q.float().square().mean(-1,keepdim=True)+1e-6).rsqrt().to(q.dtype)
        k = k * (k.float().square().mean(-1,keepdim=True)+1e-6).rsqrt().to(k.dtype)
        y = F.scaled_dot_product_attention(q,k,v)
        return self.out(y.transpose(1,2).reshape(b,n,d))


class AdaLNBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6)
        self.norm2 = nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6)
        self.attn = Attention(dim,heads)
        self.mlp = nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(approximate="tanh"),nn.Linear(dim*4,dim))
        self.mod = nn.Sequential(nn.SiLU(),nn.Linear(dim,6*dim))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, conditioning):
        s1,c1,g1,s2,c2,g2 = self.mod(conditioning).unsqueeze(1).chunk(6,-1)
        x = x + g1*self.attn(self.norm1(x)*(1+c1)+s1)
        return x + g2*self.mlp(self.norm2(x)*(1+c2)+s2)
