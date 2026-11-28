"""OpenCV camera-z depth, world-from-camera poses and deterministic z buffering."""
import torch
import torch.nn.functional as F


def unproject(depth, intrinsics, camera_to_world):
    """C,H,W axial depth to C,H,W,3 world points, retaining all six pose DoFs."""
    c,h,w=depth.shape
    y,x=torch.meshgrid(torch.arange(h,device=depth.device),torch.arange(w,device=depth.device),indexing='ij')
    pixels=torch.stack((x,y,torch.ones_like(x)),-1).to(depth.dtype)
    rays=torch.einsum('cij,hwj->chwi',torch.linalg.inv(intrinsics),pixels)
    local=rays*depth[...,None]
    return torch.einsum('cij,chwj->chwi',camera_to_world[:,:3,:3],local)+camera_to_world[:,None,None,:3,3]


def rasterize(points, colors, target_pose, target_k, size, valid=None):
    """Nearest-pixel splat; closest positive z wins, exact ties use input order.

    Returns rough RGB, M (True=observed/valid), axial depth. Holes stay black/inf.
    Does not infer hidden surfaces, blend seams or expand splat footprints.
    """
    w,h=size;points=points.reshape(-1,3);colors=colors.reshape(-1,3)
    if len(points)!=len(colors):raise ValueError('Point/color count mismatch')
    xyz=(points-target_pose[:3,3])@target_pose[:3,:3]
    uvz=xyz@target_k.T
    z=xyz[:,2];uv=uvz[:,:2]/z[:,None]
    good=torch.isfinite(xyz).all(-1)&torch.isfinite(uv).all(-1)&torch.isfinite(colors).all(-1)&(z>.05)
    if valid is not None:good&=valid.reshape(-1)
    # Clamp conversion inputs only; invalid values remain excluded by good.
    ij=torch.floor(torch.nan_to_num(uv,nan=-1e6,posinf=1e6,neginf=-1e6).clamp(-1e6,1e6)+.5).long()
    good&=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
    ids=torch.arange(len(points),device=points.device)[good];pixel=ij[good,1]*w+ij[good,0]
    nearest=torch.full((w*h,),torch.inf,device=points.device,dtype=points.dtype)
    nearest.scatter_reduce_(0,pixel,z[good],reduce='amin',include_self=True)
    visible=z[good]==nearest[pixel]
    winner=torch.full((w*h,),len(points),device=points.device,dtype=torch.long)
    winner.scatter_reduce_(0,pixel[visible],ids[visible],reduce='amin',include_self=True)
    mask=winner<len(points);rgb=torch.zeros((w*h,3),device=colors.device,dtype=colors.dtype)
    rgb[mask]=colors[winner[mask]]
    return rgb.reshape(h,w,3),mask.reshape(h,w),nearest.reshape(h,w)


def project(rgb, depth, intrinsics, camera_to_world, target_pose, target_k, size):
    if rgb.shape!=(*depth.shape,3):raise ValueError('Expected C,H,W RGB/depth')
    valid=torch.isfinite(depth)&(depth>.05)&(depth<120)
    points=unproject(torch.where(valid,depth,torch.zeros_like(depth)),intrinsics,camera_to_world)
    return rasterize(points,rgb,target_pose,target_k,size,valid)


def composite_live_first(current, current_mask, past, past_mask):
    """Fill current holes only; never replace a covered live pixel with history.

    This does not establish that historical content in current holes is still
    valid. Disoccluded moving objects can leave stale content there.
    """
    if current.shape!=past.shape or current.shape!=(*current_mask.shape,3) or past_mask.shape!=current_mask.shape:
        raise ValueError('RGB/mask dimensions must match')
    if current_mask.dtype!=torch.bool or past_mask.dtype!=torch.bool:
        raise ValueError('Coverage masks must be boolean')
    valid=current_mask|past_mask
    rgb=torch.where(current_mask[...,None],current,past)
    return torch.where(valid[...,None],rgb,torch.zeros_like(rgb)),valid


def pinhole_close_once(rgb,mask,minimum_neighbors=4):
    """One snapshot-based3x3 pass; preserve all original covered pixels exactly.

    Returned mask includes averaged small-hole pixels. Keep the original mask
    separately to distinguish observation from interpolation. Large-hole interiors
    remain empty; eligible concave boundary pixels can be filled by one pixel.
    """
    if rgb.shape!=(*mask.shape,3) or mask.dtype!=torch.bool:raise ValueError('Expected H,W,3 RGB and boolean coverage')
    if not 1<=minimum_neighbors<=8:raise ValueError('Expected1..8neighbors')
    kernel=rgb.new_ones(1,1,3,3);kernel[0,0,1,1]=0
    count=F.conv2d(mask.to(rgb.dtype)[None,None],kernel,padding=1)[0,0]
    valid_rgb=torch.where(mask[...,None],rgb,torch.zeros_like(rgb))
    sums=F.conv2d(valid_rgb.permute(2,0,1)[None],kernel.expand(3,1,3,3),padding=1,groups=3)[0].permute(1,2,0)
    filled=(~mask)&(count>=minimum_neighbors)
    closed=mask|filled
    result=torch.where(filled[...,None],sums/count.clamp_min(1)[...,None],valid_rgb)
    return result,closed,filled


def lidar_anchor(depth, sparse_depth, minimum=30):
    """Per-camera robust scale from visible nearest LiDAR samples; no GT novel view."""
    scales=[];counts=[]
    for predicted,metric in zip(depth,sparse_depth):
        ok=torch.isfinite(metric)&(metric>.5)&(metric<80)&torch.isfinite(predicted)&(predicted>.5)
        count=int(ok.sum());counts.append(count)
        scales.append((metric[ok]/predicted[ok]).median().clamp(.25,4) if count>=minimum else predicted.new_tensor(1.))
    scale=torch.stack(scales)
    return depth*scale[:,None,None],scale,counts


def align_camera_scales(depth,k,poses,minimum=30):
    """Approximate overlap log-scale graph; metric front camera is fixed gauge.

    Uses no LiDAR. Depth disagreements/occlusions and camera translation bias this
    heuristic; it is not a metric calibration guarantee. Missing edges stay prior.
    """
    c,h,w=depth.shape;sample=depth[:,::8,::8];small=k.clone()
    small[:,:2,:]/=8
    world=unproject(sample,small,poses).reshape(c,-1,3);rows=[];values=[];counts=[]
    for a in range(c):
        for b in range(a+1,c):
            xyz=(world[a]-poses[b,:3,3])@poses[b,:3,:3];uvz=xyz@k[b].T
            uv=torch.floor(uvz[:,:2]/xyz[:,2,None]+.5).long()
            ok=torch.isfinite(xyz).all(-1)&(xyz[:,2]>.5)&(sample[a].flatten()>.5)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            index=uv[ok];z=xyz[ok,2];other=depth[b,index[:,1],index[:,0]]
            ratio=z/other;valid=torch.isfinite(ratio)&(ratio>.25)&(ratio<4)
            if int(valid.sum())<minimum:continue
            row=depth.new_zeros(c);row[b]=1;row[a]=-1
            rows.append(row);values.append(ratio[valid].log().median());counts.append(dict(a=a,b=b,samples=int(valid.sum())))
    # Small identity prior handles disconnected cameras without invented edges.
    matrix=torch.cat((torch.eye(c,device=depth.device)*.1,torch.stack(rows) if rows else depth.new_zeros((0,c))),0)
    target=torch.cat((depth.new_zeros(c),torch.stack(values) if values else depth.new_zeros(0)))
    matrix=torch.cat((matrix,torch.eye(c,device=depth.device)[:1]*10));target=torch.cat((target,depth.new_zeros(1)))
    logs=torch.linalg.lstsq(matrix,target[:,None]).solution[:,0];logs=logs-logs[0]
    scales=logs.exp().clamp(.25,4)
    return depth*scales[:,None,None],scales,counts
