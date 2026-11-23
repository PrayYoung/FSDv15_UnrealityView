from contextlib import nullcontext
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


def directional_derivative(model, z, r, t, velocity, previous, tokens, ages, method="auto", epsilon=1e-3):
    """Stop-gradient derivative D_(velocity,0,1) u, no higher-order graph."""
    if method == "auto":
        half_autocast = torch.is_autocast_enabled(z.device.type) and torch.get_autocast_dtype(z.device.type)==torch.float16
        method = "finite" if z.device.type == "mps" or z.dtype == torch.float16 or half_autocast else "exact"
    with torch.no_grad():
        if method == "finite":
            # Compute differences in FP32, including on MPS/FP16 training. Casting
            # the residual to FP32 only after subtraction would lose the signal.
            with torch.autocast(z.device.type,enabled=False):
                args = [a.float() for a in (z,r,t,velocity,previous,tokens,ages)]
                zz,rr,tt,vv,pp,cc,aa = args
                plus = model(zz+epsilon*vv,rr,tt+epsilon,pp,cc,aa)
                minus = model(zz-epsilon*vv,rr,tt-epsilon,pp,cc,aa)
            return (plus-minus)/(2*epsilon)
        if method != "exact":
            raise ValueError("JVP method must be auto, exact, or finite")
        # Fused attention kernels do not implement forward-mode AD. The primal
        # training pass below retains automatic fused SDPA selection.
        with sdpa_kernel(SDPBackend.MATH):
            _, tangent = torch.func.jvp(lambda zz,tt: model(zz,r,tt,previous,tokens,ages),
                                        (z,t),(velocity,torch.ones_like(t)))
        return tangent.detach()


def meanflow_loss(model, z0, z_prev, tokens, ages=None, fm_ratio=.5,
                  prev_chunk_noise_std_max=.20, jvp_method="auto", generator=None):
    b = len(z0)
    ages = z0.new_zeros(b,4) if ages is None else ages
    t = torch.rand(b,device=z0.device,generator=generator).clamp_min(.001)
    r = torch.rand(b,device=z0.device,generator=generator)*t
    boundary = torch.rand(b,device=z0.device,generator=generator) < fm_ratio
    r = torch.where(boundary,t,r)
    eps = torch.randn(z0.shape,device=z0.device,dtype=z0.dtype,generator=generator)
    expand = lambda a: a.reshape(b,1,1,1,1)
    z = (1-expand(t))*z0+expand(t)*eps
    velocity = eps-z0
    std = torch.rand(b,device=z0.device,generator=generator)*prev_chunk_noise_std_max
    prev = z_prev+expand(std)*torch.randn(z_prev.shape,device=z_prev.device,dtype=z_prev.dtype,generator=generator)
    target = velocity.detach().clone()
    interior = ~boundary
    if interior.any():
        tangent = directional_derivative(model,z[interior],r[interior],t[interior],
                                         velocity[interior],prev[interior],tokens[interior],
                                         ages[interior],jvp_method)
        target[interior] -= expand(t-r)[interior]*tangent.to(target)
    prediction = model(z,r,t,prev,tokens,ages)
    mse = (prediction.float()-target.float().detach()).square().flatten(1).mean(1)
    weight = (mse.detach()+1e-3).pow(-.5)
    return (mse*weight).mean(), {"flow_mse":mse.detach().mean(),"interior_samples":int(interior.sum())}
