"""Coordinates: ego (forward, left, up); pinhole camera (right, down, forward)."""
import torch
import torch.nn.functional as F


def plucker_rays(intrinsics: torch.Tensor, camera_to_ego: torch.Tensor,
                 height: int = 4, width: int = 8,
                 image_size: tuple[int, int] = (140, 224)) -> torch.Tensor:
    """Return (..., height*width, 6), ordered as (direction, origin × direction)."""
    if intrinsics.shape[-2:] != (3, 3) or camera_to_ego.shape[-2:] != (4, 4):
        raise ValueError("Expected (...,3,3) intrinsics and (...,4,4) camera-to-ego")
    ih, iw = image_size
    y, x = torch.meshgrid((torch.arange(height, device=intrinsics.device) + .5) * ih / height,
                          (torch.arange(width, device=intrinsics.device) + .5) * iw / width,
                          indexing="ij")
    pixels = torch.stack((x, y, torch.ones_like(x)), -1).reshape(-1, 3).to(intrinsics)
    camera_dirs = pixels @ torch.linalg.inv(intrinsics).transpose(-1, -2)
    directions = F.normalize(camera_dirs @ camera_to_ego[..., :3, :3].transpose(-1, -2), dim=-1)
    origins = camera_to_ego[..., None, :3, 3].expand_as(directions)
    return torch.cat((directions, torch.cross(origins, directions, dim=-1)), dim=-1)


def ego_waypoints(poses: torch.Tensor, index: int, count: int = 12, stride: int = 3) -> torch.Tensor:
    """World (x,y,yaw) -> future (forward,left,sin(dyaw),cos(dyaw))."""
    ids = (torch.arange(1, count + 1, device=poses.device) * stride + index).clamp(max=len(poses)-1)
    delta = poses[ids, :2] - poses[index, :2]
    yaw = poses[index, 2]
    c, s = yaw.cos(), yaw.sin()
    xy = torch.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), -1)
    dyaw = poses[ids, 2] - yaw
    return torch.cat((xy, dyaw.sin()[:, None], dyaw.cos()[:, None]), -1)


def ground_raycast(u: float, v: float, width: int = 320, height: int = 192,
                   fov: float = 75., camera_height: float = 3.8,
                   camera_back: float = 6.8, pitch_deg: float = 19.) -> tuple[float, float] | None:
    """Pixel -> ego ground point for the chase camera; None above horizon."""
    import math
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    right, down = (u-width/2)/focal, (v-height/2)/focal
    p = math.radians(pitch_deg)
    forward, vertical = math.cos(p)-down*math.sin(p), -math.sin(p)-down*math.cos(p)
    if vertical >= -1e-5:
        return None
    t = -camera_height / vertical
    x, y = -camera_back + t * forward, -t * right
    if not (0 <= x <= 100 and abs(y) <= 30):
        return None
    return x, y


def project_waypoints(points, width=320, height=192, fov=75., pitch_deg=19.):
    import math
    focal = width / (2*math.tan(math.radians(fov)/2))
    p = math.radians(pitch_deg)
    result = []
    for x, y, *_ in points:
        fwd, up = float(x)+6.8, -3.8
        depth = fwd*math.cos(p)-up*math.sin(p)
        down = -fwd*math.sin(p)-up*math.cos(p)
        result.append([width/2-focal*float(y)/max(depth,.01), height/2+focal*down/max(depth,.01)])
    return result
