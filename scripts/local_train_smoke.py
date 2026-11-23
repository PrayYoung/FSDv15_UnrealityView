"""Actual device autocast + finite backward check; small shape, no quality claim."""
import torch
from unrealityview.device import select_device,autocast_for,synchronize
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from unrealityview.objectives import meanflow_loss

device=select_device()
model=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=32,depth=4,num_heads=4,latent_shape=(2,4,6))).to(device)
torch.nn.init.normal_(model.output.weight,std=.01)
z=torch.randn(2,4,2,4,6,device=device)
tokens=torch.randn(2,4,192,384,device=device)
with autocast_for(device):loss,stats=meanflow_loss(model,z,z,tokens,fm_ratio=0)
loss.backward()
synchronize(device)
assert torch.isfinite(loss)
assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
print({"device":str(device),"loss":loss.item(),"finite_gradients":True,"status":"DEVICE_TRAIN_SMOKE_OK"})
