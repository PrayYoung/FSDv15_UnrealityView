"""Exercise compiled CUDA kernels before downloading the native dataset."""
import torch
from gsplat import rasterization
from pytorch3d.ops import knn_points
import nvdiffrast.torch as dr


def main():
    assert torch.cuda.is_available(), 'CUDA unavailable'
    device='cuda'
    points=torch.tensor([[[-.1,0.,2.],[.1,0.,2.]]],device=device,requires_grad=True)
    neighbors=knn_points(points,points,K=1)
    neighbors.dists.sum().backward()
    assert torch.isfinite(points.grad).all()
    means=points.detach()[0].clone().requires_grad_(True)
    quats=torch.tensor([[1.,0.,0.,0.]]*2,device=device)
    colors=torch.tensor([[1.,0.,0.],[0.,1.,0.]],device=device)
    view=torch.eye(4,device=device)[None]
    k=torch.tensor([[[20.,0.,16.],[0.,20.,16.],[0.,0.,1.]]],device=device)
    rgb,alpha,_=rasterization(means,quats,torch.full((2,3),.1,device=device),
        torch.full((2,),.8,device=device),colors,view,k,32,32,packed=False)
    assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all() and alpha.max()>0
    rgb.sum().backward();assert means.grad is not None and torch.isfinite(means.grad).all()
    context=dr.RasterizeCudaContext()
    vertices=torch.tensor([[[-.5,-.5,0.,1.],[.5,-.5,0.,1.],[0.,.5,0.,1.]]],device=device)
    triangles=torch.tensor([[0,1,2]],device=device,dtype=torch.int32)
    raster,_=dr.rasterize(context,vertices,triangles,resolution=[32,32])
    assert torch.isfinite(raster).all() and raster[...,3].max()>0
    torch.cuda.synchronize()
    print('NATIVE_CUDA_FORWARD_BACKWARD_OK',flush=True)


if __name__=='__main__':main()
