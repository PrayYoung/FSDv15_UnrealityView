"""Bounded CUDA regression/performance check; random tensors are not quality evidence."""
import gc
import math
import time
import torch
from benchmark_edge import measure
from unrealityview.data.cache import atomic_json
from unrealityview.device import autocast_for
from unrealityview.models.pretrained_taesd import PretrainedTAESD
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.driving_policy_head import DrivingPolicyHead
from unrealityview.models.lora import inject_dit_lora
from unrealityview.objectives import meanflow_loss
from unrealityview.training import enable_training_backends


def main():
    device=torch.device('cuda');assert torch.cuda.is_available();enable_training_backends(device)
    result={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'note':'Software and capacity check, not trained quality evidence'}
    result['fast_decoder_dtype']='float16'
    decoder=PretrainedTAESD().to(device);z=torch.randn(1,4,8,24,40,device=device)
    with torch.no_grad():
        result['fast_batched_decode']=measure(lambda:decoder.decode(z),device,5,2)
        fast=decoder.decode(z).float();decoder.precise=True
        result['strict_batched_decode']=measure(lambda:decoder.decode(z),device,5,2)
        strict=decoder.decode(z)
        result['fast_vs_strict_pixel_l1']=float((fast-strict).abs().mean())
        atomic_json("outputs/upgrade_check/cuda_checks.json",result)
        # Historical reference also ran one frame at a time in strict FP32.
        def historical():return torch.cat([decoder.decode(z[:,:,i:i+1]) for i in range(8)],2)
        result['strict_framewise_decode']=measure(historical,device,5,2)
        small=z[:,:,:1];full=decoder.decode(small);tiled=decoder.decode_tiled(small)
        torch.testing.assert_close(full,tiled,atol=5e-5,rtol=1e-4)
        result['strict_tiled_max_error']=float((full-tiled).abs().max())
        result['fast_vs_strict_max_error']=float((fast-strict).abs().max())
        result['decode_speedup']=result['strict_framewise_decode']['median_ms']/result['fast_batched_decode']['median_ms']
    atomic_json('outputs/upgrade_check/cuda_checks.json',result)
    del decoder,z,fast,strict,full,tiled;gc.collect();torch.cuda.empty_cache()
    model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=64,depth=4,num_heads=4,latent_shape=(2,4,6))).to(device)
    inject_dit_lora(model,rank=8,start_block=0,early_rank=4)
    z=torch.randn(2,4,2,4,6,device=device);tokens=torch.randn(2,4,192,384,device=device)
    with autocast_for(device):loss,_=meanflow_loss(model,z,z,tokens,fm_ratio=0,jvp_method='exact')
    loss.backward();assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result['lora_exact_jvp_backward']=True
    del model,z,tokens,loss;gc.collect();torch.cuda.empty_cache()
    result['full_model_capacity']={}
    for batch in (16,8,4,1):
        model=ConditionalMeanFlowDiT().to(device);policy=DrivingPolicyHead().to(device)
        optimizer=torch.optim.AdamW(list(model.parameters())+list(policy.parameters()),lr=1e-4)
        z=torch.randn(batch,4,8,24,40,device=device);tokens=torch.randn(batch,4,192,384,device=device)
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        try:
            with autocast_for(device):
                loss,_=meanflow_loss(model,z,z,tokens,jvp_method='auto')
                loss=loss+policy(tokens[:,0]).square().mean()
            loss.backward();optimizer.step();torch.cuda.synchronize()
            assert torch.isfinite(loss)
            result['full_model_capacity'][str(batch)]=dict(fits=True,peak_mib=torch.cuda.max_memory_allocated()/2**20,seconds=time.monotonic()-started)
        except torch.cuda.OutOfMemoryError:
            result['full_model_capacity'][str(batch)]=dict(fits=False,reason='CUDA out of memory; capacity measurement')
        del model,policy,optimizer,z,tokens
        if 'loss' in locals():del loss
        gc.collect();torch.cuda.empty_cache()
        if result['full_model_capacity'][str(batch)]['fits']:break
    assert any(v['fits'] for v in result['full_model_capacity'].values())
    atomic_json('outputs/upgrade_check/cuda_checks.json',result)
    # Minimal gsplat kernel + gradients. Full real-scene teacher quality is separate.
    from gsplat import rasterization
    means=torch.tensor([[0.,0.,3.],[.5,0,4.]],device=device,requires_grad=True)
    colors=torch.full((2,3),.7,device=device,requires_grad=True)
    pixels,alpha,_=rasterization(means=means,quats=torch.tensor([[1.,0,0,0]],device=device).repeat(2,1),
        scales=torch.full((2,3),.2,device=device),opacities=torch.full((2,),.8,device=device),colors=colors,
        viewmats=torch.eye(4,device=device)[None],Ks=torch.tensor([[[40.,0,16],[0,40,16],[0,0,1]]],device=device),width=32,height=32,packed=False)
    pixels.mean().backward();assert torch.isfinite(means.grad).all() and torch.isfinite(colors.grad).all()
    result['gsplat_raster_backward']=True
    atomic_json('outputs/upgrade_check/cuda_checks.json',result);print(result,flush=True)


if __name__=='__main__':main()
