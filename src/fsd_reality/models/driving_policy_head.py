import torch
from torch import nn
from .blocks import Attention


class DrivingPolicyHead(nn.Module):
    """Independent transformer. Goal coordinates are meters in ego (forward,left)."""
    def __init__(self, context_dim=384, hidden_dim=288, num_heads=8, depth=3, num_waypoints=12):
        super().__init__()
        self.config = dict(context_dim=context_dim,hidden_dim=hidden_dim,num_heads=num_heads,
                           depth=depth,num_waypoints=num_waypoints)
        self.project = nn.Linear(context_dim,hidden_dim)
        self.queries = nn.Parameter(torch.randn(1,num_waypoints,hidden_dim)*.02)
        self.goal = nn.Sequential(nn.Linear(2,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim*2))
        self.layers = nn.ModuleList([nn.ModuleList([nn.LayerNorm(hidden_dim),Attention(hidden_dim,num_heads),
                                    nn.LayerNorm(hidden_dim),nn.Sequential(nn.Linear(hidden_dim,hidden_dim*4),
                                    nn.GELU(),nn.Linear(hidden_dim*4,hidden_dim))]) for _ in range(depth)])
        self.out = nn.Linear(hidden_dim,4)

    def forward(self, tokens, tap_goal_xy=None):
        b = len(tokens)
        present=tokens.detach().ne(0).any(-1)
        if tap_goal_xy is None:
            tap_goal_xy = tokens.new_zeros(b,2)
        goal = tap_goal_xy/50
        scale,shift = self.goal(goal).chunk(2,-1)
        x = self.project(tokens)
        q = self.queries.expand(b,-1,-1)*(1+scale[:,None])+shift[:,None]
        x = torch.cat((q,x),1)
        key_mask=torch.cat((torch.ones(b,q.shape[1],dtype=torch.bool,device=x.device),present),1)
        for norm,attn,norm2,mlp in self.layers:
            x = x+attn(norm(x),key_mask=key_mask)
            x = x+mlp(norm2(x))
        out = self.out(x[:,:q.shape[1]])
        heading = torch.nn.functional.normalize(out[...,2:]+1e-6,dim=-1)
        return torch.cat((out[...,:2],heading),-1)


def training_goals(waypoints, generator=None):
    goal = waypoints[:,-1,:2].clone()
    goal += torch.randn(goal.shape,device=goal.device,dtype=goal.dtype,generator=generator)*.3
    keep = torch.rand((len(goal),1),device=goal.device,generator=generator) >= .5
    return goal*keep
