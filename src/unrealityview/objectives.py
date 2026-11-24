from contextlib import contextmanager
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


def jvp_backend(z, method):
    if method == 'auto':
        return 'finite' if z.device.type == 'mps' else 'exact'
    if method not in ('exact', 'finite'):
        raise ValueError('JVP method must be auto, exact, or finite')
    return method


@contextmanager
def jvp_precision():
    matmul=torch.backends.cuda.matmul.allow_tf32;conv=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=matmul;torch.backends.cudnn.allow_tf32=conv


def directional_derivative(model, z, r, t, velocity, previous, tokens, ages, method='auto', epsilon=1e-3):
    """FP32 stop-gradient D_(velocity,0,1) u; never retain a higher-order graph."""
    method = jvp_backend(z, method)
    with torch.no_grad(), torch.autocast(z.device.type, enabled=False), jvp_precision():
        z,r,t,velocity,previous,tokens,ages = [x.float() for x in (z,r,t,velocity,previous,tokens,ages)]
        if method == 'finite':
            if epsilon <= 0:raise ValueError('epsilon must be positive')
            # Keep probes inside [r,1], including the corner r=t or t=1.
            lo = torch.minimum(t-r, torch.full_like(t, epsilon)).clamp_min(0)
            hi = torch.minimum(1-t, torch.full_like(t, epsilon)).clamp_min(0)
            if ((lo+hi)==0).any():raise ValueError('No finite-difference interval at r=t=1')
            expand = lambda a:a.reshape(-1,1,1,1,1)
            plus = model(z+expand(hi)*velocity,r,t+hi,previous,tokens,ages)
            minus = model(z-expand(lo)*velocity,r,t-lo,previous,tokens,ages)
            return (plus-minus)/expand(lo+hi)
        with sdpa_kernel(SDPBackend.MATH):
            _, tangent = torch.func.jvp(lambda zz,tt:model(zz,r,tt,previous,tokens,ages),
                (z,t),(velocity,torch.ones_like(t)))
        return tangent.detach()


def sample_times(batch,device,mode,generator=None):
    if mode == 'logit_normal':
        values=torch.sigmoid(torch.randn(2,batch,device=device,generator=generator)-.4).clamp(.001,1)
        return values.min(0).values,values.max(0).values
    if mode == 'uniform':
        t=torch.rand(batch,device=device,generator=generator).clamp_min(.001)
        return torch.rand(batch,device=device,generator=generator)*t,t
    raise ValueError('time_sampling must be logit_normal or uniform')


def adaptive_loss(mse,p=1.,c=.01):
    if not 0<=p<=1 or c<=0:raise ValueError('adaptive_p must be in [0,1] and adaptive_c positive')
    # FP64 protects squaring/weighting extreme finite synthetic residuals.
    return (mse*(mse.detach()+c).pow(-p)).mean().float()


def meanflow_loss(model,z0,z_prev,tokens,ages=None,fm_ratio=.75,
                  prev_chunk_noise_std_max=.20,jvp_method='auto',generator=None,
                  adaptive_p=1.,adaptive_c=.01,time_sampling='logit_normal',jvp_weight=1.):
    if not 0<=fm_ratio<=1 or not 0<=jvp_weight<=1 or prev_chunk_noise_std_max<0:
        raise ValueError('Invalid Mean-Flow configuration')
    b=len(z0);ages=z0.new_zeros(b,4) if ages is None else ages
    r,t=sample_times(b,z0.device,time_sampling,generator)
    boundary=torch.rand(b,device=z0.device,generator=generator)<fm_ratio
    r=torch.where(boundary,t,r)
    eps=torch.randn(z0.shape,device=z0.device,dtype=z0.dtype,generator=generator)
    expand=lambda a:a.reshape(b,1,1,1,1)
    z=(1-expand(t))*z0+expand(t)*eps
    velocity=eps-z0
    std=torch.rand(b,device=z0.device,generator=generator)*prev_chunk_noise_std_max
    prev=z_prev+expand(std)*torch.randn(z_prev.shape,device=z_prev.device,dtype=z_prev.dtype,generator=generator)
    target=velocity.detach().float().clone();interior=~boundary
    backend=jvp_backend(z,jvp_method);tangent=None
    if interior.any() and jvp_weight:
        tangent=directional_derivative(model,z[interior],r[interior],t[interior],velocity[interior],
            prev[interior],tokens[interior],ages[interior],backend)
        target[interior]-=jvp_weight*expand(t-r)[interior]*tangent
    prediction=model(z,r,t,prev,tokens,ages)
    mse=(prediction.float()-target.detach()).to(torch.float32 if z0.device.type=="mps" else torch.float64).square().flatten(1).mean(1)
    if not torch.isfinite(mse).all():raise RuntimeError('Nonfinite raw Mean-Flow residual')
    rms=lambda x:x.detach().to(torch.float32 if z0.device.type=="mps" else torch.float64).square().mean().sqrt()
    subset=lambda mask:mse[mask].detach().mean() if mask.any() else mse.new_zeros(())
    return adaptive_loss(mse,adaptive_p,adaptive_c),dict(flow_mse=mse.detach().mean(),
        boundary_mse=subset(boundary),interior_mse=subset(interior),
        du_dt_rms=rms(tangent) if tangent is not None else mse.new_zeros(()),
        target_rms=rms(target),pred_rms=rms(prediction),interior_samples=int(interior.sum()),
        jvp_backend=backend if tangent is not None else 'skipped',jvp_weight=float(jvp_weight))
