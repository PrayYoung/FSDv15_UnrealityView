import math
import torch
from unrealityview.explicit3d.project import project,rasterize,unproject,lidar_anchor,align_camera_scales


def test_identity_projection_preserves_rgb_depth_and_mask():
    image=torch.arange(36,dtype=torch.float32).reshape(1,3,4,3)/36
    depth=torch.full((1,3,4),5.);k=torch.tensor([[[3.,0,1.5],[0,3.,1],[0,0,1]]]);pose=torch.eye(4)[None]
    rgb,m,z=project(image,depth,k,pose,pose[0],k[0],(4,3))
    torch.testing.assert_close(rgb,image[0]);torch.testing.assert_close(z,depth[0]);assert m.all()


def test_closest_depth_wins_and_behind_or_invalid_samples_are_excluded():
    points=torch.tensor([[0.,0,4],[0,0,2],[0,0,-1],[float('nan'),0,1],[10,0,1]])
    colors=torch.eye(3)[torch.tensor([0,1,2,2,2])]
    rgb,m,z=rasterize(points,colors,torch.eye(4),torch.eye(3),(2,2))
    torch.testing.assert_close(rgb[0,0],torch.tensor([0.,1,0]));assert z[0,0]==2 and m.sum()==1
    assert not m[1,1] and torch.isinf(z[1,1])


def test_full_six_dof_pose_roundtrip_and_translation():
    # Rotation contains pitch,roll,yaw; depth is camera-z,not radial distance.
    q=torch.tensor([.8,.1,.2,.3]);q=q/q.norm();w,x,y,z=q
    r=torch.tensor([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    p=torch.eye(4);p[:3,:3]=r;p[:3,3]=torch.tensor([20.,-4,3])
    depth=torch.full((1,2,2),3.);rgb=torch.rand(1,2,2,3);k=torch.eye(3)[None]
    out,m,z=project(rgb,depth,k,p[None],p,k[0],(2,2))
    assert m.all();torch.testing.assert_close(out,rgb[0]);torch.testing.assert_close(z,depth[0],atol=1e-5,rtol=1e-5)


def test_six_camera_occlusion_and_holes_have_no_fabricated_rgb():
    rgb=torch.zeros(6,1,1,3);rgb[4,0,0]=torch.tensor([.2,.4,.8])
    depth=torch.tensor([6.,5,4,3,1,2]).reshape(6,1,1)
    out,m,z=project(rgb,depth,torch.eye(3).repeat(6,1,1),torch.eye(4).repeat(6,1,1),torch.eye(4),torch.eye(3),(2,2))
    assert m.sum()==1;torch.testing.assert_close(out[0,0],rgb[4,0,0]);assert out[1,1].sum()==0


def test_lidar_scales_and_insufficient_sample_fallback():
    depth=torch.full((2,8,8),10.);sparse=torch.full_like(depth,20.);sparse[1]=torch.inf
    out,scale,count=lidar_anchor(depth,sparse)
    torch.testing.assert_close(scale,torch.tensor([2.,1.]));assert count==[64,0];assert out[0,0,0]==20


def test_camera_only_scale_alignment_uses_overlap_and_front_gauge():
    depth=torch.stack((torch.full((64,64),5.),torch.full((64,64),10.)))
    out,scale,edges=align_camera_scales(depth,torch.eye(3).repeat(2,1,1),torch.eye(4).repeat(2,1,1))
    assert len(edges)==1 and scale[0]==1 and abs(float(scale[1])-.5)<.01
    torch.testing.assert_close(out[0],out[1],atol=.05,rtol=.01)
