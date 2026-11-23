"""Exercise the actual CUDA autocast/JVP path before paying for data collection."""
import torch
from unrealityview.device import autocast_for
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.objectives import meanflow_loss

device=torch.device("cuda")
model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=64,depth=4,num_heads=4,latent_shape=(2,4,6))).to(device)
torch.nn.init.normal_(model.output.weight,std=.01)
z=torch.randn(2,4,2,4,6,device=device)
tokens=torch.randn(2,4,192,384,device=device)
for method in ("auto","finite"):
    with autocast_for(device):
        loss,_=meanflow_loss(model,z,z,tokens,fm_ratio=0,jvp_method=method)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
print("CUDA_AUTOCAST_JVP_BACKWARD_OK",flush=True)
