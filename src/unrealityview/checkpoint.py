import json
from pathlib import Path
from safetensors.torch import load_file, save_file
from unrealityview.data.cache import atomic_json
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.models.driving_policy_head import DrivingPolicyHead


def save_checkpoint(directory, model, policy, step, extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    tensors = {"dit."+k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    tensors.update({"policy."+k:v.detach().cpu().contiguous() for k,v in policy.state_dict().items()})
    temporary = directory/"model.safetensors.tmp"
    save_file(tensors,str(temporary),metadata={"step":str(step),"format":"unrealityview-v1"})
    temporary.replace(directory/"model.safetensors")
    atomic_json(directory/"config.json",{"dit":model.config.to_dict(),"policy":policy.config,"step":step,**(extra or {})})


def load_checkpoint(directory, device="cpu"):
    directory = Path(directory)
    config = json.loads((directory/"config.json").read_text())
    config["dit"].setdefault("cross_attention","stage")
    config["dit"].setdefault("time_scale",1000.0)
    model = ConditionalMeanFlowDiT(DiTConfig(**config["dit"]))
    policy = DrivingPolicyHead(**config["policy"])
    weights = load_file(str(directory/"model.safetensors"))
    model.load_state_dict({k[4:]:v for k,v in weights.items() if k.startswith("dit.")},strict=True)
    policy.load_state_dict({k[7:]:v for k,v in weights.items() if k.startswith("policy.")},strict=True)
    return model.to(device).eval(),policy.to(device).eval(),config
