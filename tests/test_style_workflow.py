"""Exercise shader cache -> one optimizer step -> safe export with test fixtures.

These fixtures and outputs are never model-quality or public demo evidence.
"""
import hashlib
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from unrealityview.checkpoint import save_checkpoint,load_checkpoint
from unrealityview.data.cache import save_episode
from unrealityview.data.stylize_cache import build_style_cache
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.driving_policy_head import DrivingPolicyHead
from unrealityview.models.lora import load_lora,model_fingerprint


class FixtureCodec(torch.nn.Module):
    def encode(self,pixels):
        rgb=F.interpolate(pixels[:,:,0],size=(24,40),mode="bilinear",align_corners=False)
        return torch.cat((rgb,rgb.mean(1,keepdim=True)),1)[:,:,None]

    def decode(self,latents):
        b,c,t,h,w=latents.shape
        rgb=latents[:,:3].permute(0,2,1,3,4).reshape(b*t,3,h,w)
        return F.interpolate(rgb,size=(32,48),mode="bilinear",align_corners=False).reshape(b,t,3,32,48).permute(0,2,1,3,4)


def test_cache_training_export_and_internal_evaluation(tmp_path,monkeypatch):
    import train_style_lora
    data=tmp_path/"base";styles=tmp_path/"styles";hashes={}
    for split in ("train","val"):
        directory=data/split/"fixture"
        save_episode(directory,{"uve_tokens":torch.randn(64,192,384),
            "chase_latents":torch.zeros(64,4,24,40),"ego_states":torch.zeros(64,3)},
            {"source":"pandaset","source_id":f"pandaset/unit-fixture-{split}","fps":10.,"frames":64,"target_view":"native_front"})
        raw=directory/"raw/chase";raw.mkdir(parents=True)
        image=np.zeros((32,48,3),dtype=np.uint8);image[:16]=[70,130,200];image[16:]=[60,60,60]
        for index in range(64):Image.fromarray(np.roll(image,index%4,axis=1)).save(raw/f"{index:05d}.jpg")
        path=directory/"episode.safetensors";hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
    build_style_cache(data,styles,("comic",),"cpu",FixtureCodec())
    for path,digest in hashes.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    base=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4))
    torch.nn.init.normal_(base.output.weight,std=.01)
    policy=DrivingPolicyHead(hidden_dim=16,depth=1,num_heads=4)
    checkpoint=tmp_path/"checkpoint";save_checkpoint(checkpoint,base,policy,1)
    before=model_fingerprint(policy)
    monkeypatch.setattr(train_style_lora,"PretrainedTAESD",FixtureCodec)
    args=SimpleNamespace(device="cpu",seed=1,base=checkpoint,rank=4,alpha=4.,start_block=0,no_unpatch=False,
        data=data,styles_data=styles,style="comic",batch_size=1,lr=2.5e-4,output=tmp_path/"training",
        grad_accum=1,steps=2,stop_after=1,warmup=1,ema=.9,eval_every=1,eval_items=1)
    train_style_lora.train(args)
    args.resume=args.output/"checkpoint";args.stop_after=None
    train_style_lora.train(args)
    assert (args.output/"checkpoint_best/comic_lora.safetensors").is_file()
    assert json.loads((args.output/"summary.json").read_text())["steps"]==2
    adapter=load_lora(args.output/"comic_lora.safetensors",model_fingerprint(base))
    assert any(t.abs().sum()>0 for key,t in adapter.tensors.items() if key.endswith("lora_b"))
    assert json.loads((args.output/"summary.json").read_text())["policy_unchanged"] is True
    assert (args.output/"eval_000001/closed_loop.gif").is_file()
    _,after,_=load_checkpoint(checkpoint)
    assert model_fingerprint(after)==before
