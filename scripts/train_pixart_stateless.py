"""Prompt D: one budgeted stateless real-PandaSet pilot, no archive access."""
import argparse,json,math,time,random,os
from pathlib import Path
import torch
from safetensors.torch import save_file
from diffusers import PixArtTransformer2DModel,AutoencoderTiny
from unrealityview.pipeline2.model import StatelessPixArt
from unrealityview.pipeline2.data import RecoveryScenes
from unrealityview.pipeline2.losses import ChannelNormalizedVGG16,three_term_loss
from unrealityview.data.cache import atomic_json
from unrealityview.runtime.optimizer_progress import record_optimizer_progress


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--arm-e',required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--grad-accum',type=int,default=2);a=p.parse_args()
    if a.steps!=2000 or a.batch_size*a.grad_accum!=16:raise ValueError('Frozen pilot requires2000updates/effectivebatch16')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.manual_seed(42);random.seed(42);torch.set_num_threads(min(8,os.cpu_count() or 1))
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    device='cuda';output=Path(a.output);output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    os.environ['UNREALITY_PROGRESS_FILE']=str(output/'optimizer_progress.json')
    pins=json.loads(Path('configs/pixart_stateless_pins.json').read_text())
    atomic_json(output/'run_config.json',dict(steps=2000,effective_batch=16,batch_size=a.batch_size,grad_accum=a.grad_accum,
        seed=42,pins=pins,camera_lag=1,token_count=1200,stateless=True,latent_history=False,
        lora_rank=16,lora_alpha=32,learning_rate=1.5e-4,warmup=100,ema_decay=.992,
        loss_weights=dict(pseudo_huber=.5,charbonnier=1.,channel_normalized_vgg16=.8),
        augmentation='two independently RGB-jittered cached banks,selected per camera per sample;single-camera dropout0.10',
        sdpa_backend='FlashAttention CUDA only',normalized_time=1.,pretrained_timestep=999,
        expected_train_scenes=60,expected_val_scenes=8,reserved_test_access=False))
    train=RecoveryScenes(a.cache,'train',60);val=RecoveryScenes(a.cache,'val',8)
    if set(train.ids)&set(val.ids):raise ValueError('Train/validation leakage')
    atomic_json(output/'data_receipt.json',dict(train=train.ids,val=val.ids,target='native_front',camera_frames='t-1',target_frames='t',frames_per_scene=80))
    ae=AutoencoderTiny.from_pretrained(pins['taesd']['repo'],revision=pins['taesd']['revision'],use_safetensors=True).to(device).eval().requires_grad_(False)
    encoded=time.monotonic();train.encode(ae,device);val.encode(ae,device)
    atomic_json(output/'preencode.json',dict(seconds=time.monotonic()-encoded,train_banks=3,val_banks=1,archive_download=False))
    base=PixArtTransformer2DModel.from_pretrained(pins['transformer']['repo'],revision=pins['transformer']['revision'],subfolder='transformer',use_safetensors=True,torch_dtype=torch.bfloat16)
    if len(base.transformer_blocks)!=28:raise ValueError('Expected pretrained28blockbackbone')
    model=StatelessPixArt(base).to(device,dtype=torch.bfloat16)
    for parameter in model.parameters():
        if parameter.requires_grad:parameter.data=parameter.data.float()
    params={n:p for n,p in model.named_parameters() if p.requires_grad};ema={n:p.detach().clone() for n,p in params.items()}
    atomic_json(output/'parameters.json',dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in params.values()),trainable_names=list(params)))
    optimizer=torch.optim.AdamW(params.values(),lr=1.5e-4,weight_decay=.01);vgg=ChannelNormalizedVGG16().to(device).eval()
    model.train();curve=[];fit_start=time.monotonic()
    for step in range(a.steps):
        scale=min(1.,(step+1)/100)*.5*(1+math.cos(math.pi*max(0,step-100)/1900))
        for group in optimizer.param_groups:group['lr']=1.5e-4*scale
        optimizer.zero_grad(set_to_none=True);totals={}
        for micro in range(a.grad_accum):
            pairs=[(random.randrange(60),random.randrange(1,80)) for _ in range(a.batch_size)]
            cams,camr,tr,z,gt=train.batch(pairs,device,train=True)
            noise=torch.randn_like(z)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(noise,cams,camr,tr);pixels=(ae.decode(pred).sample+1)/2
                loss,terms=three_term_loss(pred,z,pixels,gt,vgg)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            (loss/a.grad_accum).backward()
            for key,value in {'total':loss,**terms}.items():totals[key]=totals.get(key,0.)+float(value.detach())/a.grad_accum
        norm=torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():
            for name,param in params.items():ema[name].lerp_(param,.008)
        record_optimizer_progress(step,totals,note='Written after completed AdamW optimizer.step')
        if step%20==0 or step==a.steps-1:
            row=dict(step=step+1,seconds=time.monotonic()-fit_start,gradient_norm=float(norm),lr=optimizer.param_groups[0]['lr'],**totals);curve.append(row);atomic_json(output/'training_curve.json',curve)
    save_file({k:v.float().cpu().contiguous() for k,v in ema.items()},str(output/'adapter_ema.safetensors'))
    with torch.no_grad():
        for name,param in params.items():param.copy_(ema[name])
    del optimizer,ema,params;torch.cuda.empty_cache()
    model.eval().fuse_lora()
    from unrealityview.pipeline2.evaluate import evaluate_pilot
    evaluate_pilot(model,ae,vgg,val,device,output,Path(a.arm_e))
    atomic_json(output/'summary.json',dict(status='completed',completed_updates=a.steps,fit_seconds=time.monotonic()-fit_start,total_seconds=time.monotonic()-start))

if __name__=='__main__':main()
